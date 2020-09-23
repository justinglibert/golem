import reverb
print("trying to connect...")
client = reverb.Client('172.31.45.61:8000')
print(client.server_info())
