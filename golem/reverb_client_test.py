import reverb
print("trying to connect...")
client = reverb.Client('172.31.45.61:8000')
print(client.server_info())

for i in range(100):
    client.insert(([i,2,3, [4,5,6]], i), priorities={'my_table': 1.0})

for batch in list(client.sample('my_table', num_samples=4)):
        print("New batch")
        print(len(batch))
        print(batch[0])
        print(batch[0].data)
        print(type(batch[0].data))
        for i in batch[0].data:
            print(i)
