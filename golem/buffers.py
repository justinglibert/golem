import itertools as it
from typing import Union, Dict, List, Tuple, Any, Callable
import torch as t
import random
from copy import deepcopy
import torch
from threading import RLock
import numpy as np
from .distributed import RpcGroup


class TransitionStorage(list):
    """
    TransitionStorageBasic is a linear, size-capped chunk of memory for
    transitions, it makes sure that every stored transition is copied,
    and isolated from the passed in transition object.
    """

    def __init__(self, max_size):
        """
        Args:
            max_size: Maximum size of the transition storage.
        """
        self.max_size = max_size
        self.index = 0
        super(TransitionStorage, self).__init__()

    def store(self, og_transition: Dict) -> int:
        """
        Args:
            transition: Transition object to be stored

        Returns:
            The position where transition is inserted.
        """
        transition = {}
        for k in og_transition.keys():
            curr = og_transition[k]
            if type(curr) is torch.Tensor:
                transition[k] = og_transition[k].detach().clone()
            elif type(curr) is tuple:
                t = []
                for tensor in og_transition[k]:
                    t.append(tensor.detach().clone())
                tup = tuple(t)
                transition[k] = tup
            else:
                raise Exception(
                    "Cannot preprocess this type of transition value: {}".format(type(curr)))
        if len(self) == self.max_size:
            # ring buffer storage
            position = self.index
            self[self.index] = transition
        elif len(self) < self.max_size:
            # append if not full
            self.append(transition)
            position = len(self) - 1
        else:  # pragma: no cover
            raise RuntimeError()
        self.index = (position + 1) % self.max_size
        return position

    def clear(self):
        super(TransitionStorage, self).clear()


def _round_up(num):
    return int(np.ceil(num))


class DistributedBuffer(object):
    def __init__(self, buffer_name: str, group: RpcGroup, buffer_size: int,
                 *_, **__):
        """
        Create a distributed replay buffer instance.

        To avoid issues caused by tensor device difference, all transition
        objects are stored in device "cpu".

        Distributed replay buffer constitutes of many local buffers held per
        process, transmissions between processes only happen during sampling.

        During sampling, the tensors in "state", "action" and "next_state"
        dictionaries, along with "reward", will be concatenated in dimension 0.
        any other custom keys specified in ``**kwargs`` will not be
        concatenated.

        .. seealso:: :class:`.Buffer`

        Note:
            Since ``append()`` operates on the local buffer, in order to
            append to the distributed buffer correctly, please make sure
            that your actor is also the local buffer holder, i.e. a member
            of the ``group``

        Args:
            buffer_size: Maximum local buffer size.
            group: Process group which holds this buffer.
            buffer_name: A unique name of your buffer.
        """
        self.buffer_size = buffer_size
        self.buffer_device = 'cpu'
        self.buffer = TransitionStorage(buffer_size)
        self.index = 0
        self.buffer_name = buffer_name
        self.group = group

        assert group.is_member()

        # register services, so that we may access other buffers
        _name = "/" + group.get_cur_name()
        self.group.register(buffer_name + _name + "/_size_service",
                            self._size_service)
        self.group.register(buffer_name + _name + "/_clear_service",
                            self._clear_service)
        self.group.register(buffer_name + _name + "/_sample_service",
                            self._sample_service)
        self.wr_lock = RLock()

    @staticmethod
    def sample_method_random_unique(buffer: List[Dict], batch_size: int) \
            -> Tuple[int, List[Dict]]:
        """
        Sample unique random samples from buffer.

        Note:
            Sampled size could be any value from 0 to ``batch_size``.
        """
        if len(buffer) < batch_size:
            batch = random.sample(buffer, len(buffer))
            real_num = len(buffer)
        else:
            batch = random.sample(buffer, batch_size)
            real_num = batch_size
        return real_num, batch

    @staticmethod
    def sample_method_random(buffer: List[Dict], batch_size: int) \
            -> Tuple[int, List[Dict]]:
        """
        Sample random samples from buffer.

        Note:
            Sampled size could be any value from 0 to ``batch_size``.
        """
        indexes = [random.randint(0, len(buffer) - 1)
                   for _ in range(batch_size)]
        batch = [buffer[i] for i in indexes]
        return batch_size, batch

    @staticmethod
    def sample_method_all(buffer: List[Dict], _) \
            -> Tuple[int, List[Dict]]:
        """
        Sample all samples from buffer. Always return the whole buffer,
        will ignore the ``batch_size`` parameter.
        """
        return len(buffer), buffer

    def append(self, transition: Dict):
        with self.wr_lock:
            return self.buffer.store(transition)

    def clear(self):
        """
        Clear current local buffer.
        """
        with self.wr_lock:
            self.buffer.clear()

    def all_clear(self):
        """
        Remove all entries from all local buffers.
        """
        future = [
            self.group.registered_async(
                self.buffer_name + "/" + m + "/_clear_service"
            )
            for m in self.group.get_group_members()
        ]
        for fut in future:
            fut.wait()

    def size(self):
        """
        Returns:
            Length of current local buffer.
        """
        with self.wr_lock:
            return len(self.buffer)

    def all_size(self):
        """
        Returns:
            Total length of all buffers.
        """
        future = []
        count = 0
        for m in self.group.get_group_members():
            future.append(self.group.registered_async(
                self.buffer_name + "/" + m + "/_size_service"
            ))
        for fut in future:
            count += fut.wait()
        return count

    def sample_batch(self,
                     batch_size: int,
                     task_name: str,
                     batch_size_force_multiple: int = 1,
                     sample_method: Union[Callable, str] = "random_unique",
                     ) -> Any:
        # p_num = self.group.size()
        filtered_group = list(
            filter(lambda p: task_name in p, self.group.get_group_members()))
        p_num = len(filtered_group)
        local_batch_size = _round_up(batch_size / p_num)

        future = [
            self.group.registered_async(
                self.buffer_name + "/" + m + "/_sample_service",
                args=(local_batch_size, sample_method)
            )
            for m in filtered_group
        ]

        results = [fut.wait() for fut in future]
        all_batch_size = sum([r[0] for r in results])
        all_batch = list(it.chain(*[r[1] for r in results]))
        reminder = all_batch_size % batch_size_force_multiple
        all_batch_size -= reminder
        all_batch = all_batch[:all_batch_size]

        return all_batch_size, all_batch

    def _size_service(self):  # pragma: no cover
        return self.size()

    def _clear_service(self):  # pragma: no cover
        self.clear()

    def _sample_service(self, batch_size, sample_method):  # pragma: no cover
        if isinstance(sample_method, str):
            if not hasattr(self, "sample_method_" + sample_method):
                raise RuntimeError("Cannot find specified sample method: {}"
                                   .format(sample_method))
            sample_method = getattr(self, "sample_method_" + sample_method)

        # sample raw local batch from local buffer
        with self.wr_lock:
            local_batch_size, local_batch = sample_method(self.buffer,
                                                          batch_size)

        return local_batch_size, local_batch


class DistributedBufferTorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, buffer: DistributedBuffer, **kwargs):
        super(DistributedBufferTorchDataset, self).__init__()
        self.buffer = buffer
        self.kwargs = kwargs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            print("Running in a single process! bad.")

        return iter(lambda _: self.buffer.sample_batch(**self.kwargs), False)
