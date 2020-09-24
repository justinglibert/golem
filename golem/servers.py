
from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Lock
from copy import deepcopy
from torch.nn import functional as F
from torch import nn
import torch
from .distributed import RpcGroup, get_world, get_cur_name
from typing import Any


def prep_load_state_dict(model: nn.Module,
                         state_dict: Any):
    """
    Automatically load a **loaded state dictionary**

    Note:
        This function handles tensor device remapping.
    """
    for name, param in model.state_dict().items():
        state_dict[name].to(param.device)
    model.load_state_dict(state_dict)


class OrderedServerBase(ABC):  # pragma: no cover
    """
    Descendent classes of OrderedServer does not have to guarantee strong
    consistency, that is, even if :meth:`.OrderedServerBase.push_service``
    has returned True, there are possibilities that these acknowledged
    push are discarded.
    """
    @abstractmethod
    def push(self, key, value, version, prev_version):
        """
        Push a new ``version`` of ``value`` in ``key`` to the ordered server.

        Note:
            If ``version = prev_version`` then there is no order guarantee. But
            you may exploit this feature.

        Args:
            key: Key.
            value: value.
            version: New version.
            prev_version: Previous version.

        Returns:
            ``True`` if success, and ``False`` if not.
        """
        pass

    @abstractmethod
    def pull(self, key, version=None):
        """
        Pull a value with the specified ``version`` in ``key``.

        Args:
            key: Key.
            version: Target version, if ``None``, then the newest version
                of value of key will be pulled.

        Returns:
            ``None`` if version is not found, auto-deleted, or key is not found,
            otherwise returns value with the specified ``version``
            in ``key``, and then ``version``
        """
        pass


class OrderedServerSimple(OrderedServerBase):
    def __init__(self, server_name: str, group: RpcGroup):
        self._push_service = server_name + "/_push_service"
        self._pull_service = server_name + "/_pull_service"
        self.group = group

    def push(self, key, value, version, prev_version):
        # DOC INHERITED
        return self.group.registered_sync(
            self._push_service, args=(key, value, version, prev_version)
        )

    def pull(self, key, version=None):
        # DOC INHERITED
        return self.group.registered_sync(
            self._pull_service, args=(key, version)
        )


class OrderedServerSimpleImpl(object):
    """
    A simple key-value server, with strict ordered update
    """

    def __init__(self,
                 server_name: str,
                 group: RpcGroup,
                 version_depth: int = 1,
                 **__):
        """
        This init function must be only invoked on the runner process,
        and the runner process must be a member process of ``group``.

        Args:
            server_name: Name of this server, used to registered
                the server as a paired class of ``group``.
            group: Rpc group where server locates.
            server_runner: Name of the process serving the ordered server.
                By default is the first member of the group.
            version_depth: Storage depth of old versions of the same
                key. If ``depth = 1``, then only the newest version
                of the key will be saved.
        """
        assert group.is_member()
        assert version_depth > 0 and isinstance(version_depth, int)

        self.server_name = server_name
        self.group = group
        self.storage = {}
        self.lock = Lock()
        self.version_depth = version_depth
        # pair an accessor to group
        self.group.pair(server_name,
                        OrderedServerSimple(self.server_name, self.group))
        self.group.register(server_name + "/_push_service", self._push_service)
        self.group.register(server_name + "/_pull_service", self._pull_service)

    def _push_service(self, key, value, version, prev_version):
        success = False
        with self.lock:
            if key in self.storage:
                ref = self.storage[key]
                # Check previous version consistency.
                if next(reversed(ref)) == prev_version:
                    ref[version] = value
                    success = True
                if len(ref) > self.version_depth + 1:
                    ref.popitem(last=False)
            else:
                # Create a new key.
                ref = self.storage[key] = OrderedDict()
                ref[version] = value
                success = True
        return success

    def _pull_service(self, key, version=None):
        result = None
        with self.lock:
            if key in self.storage:
                ref = self.storage[key]
                # Try to find the target version.
                if version is not None and version in ref:
                    result = (deepcopy(ref[version]), version)
                # Find the newest version.
                elif version is None:
                    version = next(reversed(ref))
                    result = (deepcopy(ref[version]), version)
        return result


class PushPullModelServer:
    def __init__(self,
                 model_name: str,
                 o_server: OrderedServerBase = None):
        """
        Create an accessor to the services provided by
        :class:`PushPullModelServerImpl`

        Args:
            model_name: Name of the managed model in the ordered server,
                only needed if ``server`` needs such a identifier. The default
                ordered server does not require this.
            o_server: Ordered server accessor.
        """
        self.model_name = model_name
        self.o_server = o_server

    def push(self, model: nn.Module, pull_on_fail=True):
        """
        Try to push a model to the ordered server, if failed, the newest
        model will be automatically pulled and its parameters will be
        assigned to ``model``. Gradients will not be cleared.

        Args:
            model: Model to push.
            pull_on_fail: Pull the newest parameters if push failed.
        """
        if not hasattr(model, "pp_version"):
            model.pp_version = 0

        copied_model_params = deepcopy(model.state_dict())
        for k, v in copied_model_params.items():
            copied_model_params[k] = v.to("cpu")
        if not self.o_server.push(
                self.model_name, copied_model_params,
                version=model.pp_version + 1, prev_version=model.pp_version
        ):
            if pull_on_fail:
                result = self.o_server.pull(self.model_name)
                if result is None:  # pragma: no cover
                    raise RuntimeError("Pull failed, this should not happen.")
                st_dict, version = result
                prep_load_state_dict(model, st_dict)
                model.pp_version = version
            return False
        else:
            model.pp_version += 1
        return True

    def pull(self, model: nn.Module):
        """
        Pull the newest state dict of your model and update its parameters
        and ``pp_version``. Gradients will not be cleared.

        Args:
            model: Model to pull.
        """
        result = self.o_server.pull(self.model_name)
        if result is None:  # pragma: no cover
            return False
        st_dict, version = result
        if not hasattr(model, "pp_version") or model.pp_version < version:
            prep_load_state_dict(model, st_dict)
            model.pp_version = version
        return True


class PushPullModelServerImpl:
    """
    A simple parameter server, which synchronize model parameters
    by pushing and pulling all parameters and maintaining a strict
    ordered version chain.

    Warning:
        Only one model is supported.
    """

    def __init__(self,
                 server_name: str,
                 group: RpcGroup,
                 model_name: str = "model",
                 o_server: OrderedServerBase = None):
        """
        This init function must be only invoked on the runner process,
        and the runner process must be a member process of ``group``.

        Args:
            server_name: Name of this server, used to registered
                the server as a paired class of ``group``.
            group: RpcGroup of the default server :class:`.OrderedServerSimple`
                mutually exclusive with ``o_server``
            model_name: Name of the managed model in the ordered server,
                only needed if ``server`` needs such a identifier. The default
                ordered server does not require this.
            o_server: Custom ordered server accessor.
        """
        self.server_name = server_name
        self.group = group
        self.model_name = model_name
        # actual running server started by OrderedServerSimpleStarter
        self._o_server_impl = None
        if o_server is None:
            self._o_server_impl = OrderedServerSimpleImpl(
                server_name + "_o_server", group
            )
            self.o_server = group.get_paired(server_name + "_o_server")\
                                 .to_here()
        else:  # pragma: no cover
            self.o_server = o_server
        # pair an accessor to group
        self.group.pair(server_name,
                        PushPullModelServer(self.model_name, self.o_server))


def model_server_helper(model_num):
    """
    Helper function for creating a tuple of model servers,
    used by APEX, etc. This function requires all processes
    in the world to enter.

    Warning:
        You should never run this function twice!

    Returns:
        A tuple of accessors to model servers, the size of tuple is
        ``model_num``
    """
    DEFAULT_GROUP_NAME = "server_group"

    # create groups first
    world = get_world()
    server_group = world.create_rpc_group(DEFAULT_GROUP_NAME,
                                          world.get_members())

    # create servers
    # In current implementation, only one process will initialize the server
    if get_cur_name() == world.get_members()[0]:
        for i in range(model_num):
            _server = PushPullModelServerImpl("model_server_" + str(i),
                                              server_group)

    server_group.barrier()

    servers = tuple(
        server_group.get_paired("model_server_" + str(i)).to_here()
        for i in range(model_num)
    )

    # accessors instead of actual implementation instance
    # will be returned because of __reduce__
    return servers
