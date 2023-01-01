import requests

url = "http://127.0.0.1:8973/faceswap?content_image&target_image"

files=[('content_image', ('input.png', open('source_image/download.png', 'rb'), 'image/png')),
       ('target_image', ('target.jpeg', open('swap_image/00747587-e2d4-4769-9c28-bef82fa4043f.jpeg', 'rb'), 'image/jpeg'))]

response = requests.request("POST", url, files=files)
