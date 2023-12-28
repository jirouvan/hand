import requests
from PIL import Image
import io
import base64

url = "http://127.0.0.1:7860"
x = input("请输入要生成的图形信息：")
txt2img_data = {
    "prompt" : f"{x}",
    "sampler_index": "DPM++ 2M Karras",
    "batch_size": 1,
    "steps": 20,
    "cfg_scale": 7,
    "width": 800,
    "height": 800,
    "negative_prompt": "NSFW,blurry,bad anatomy,low quality,worst quality,normal quality,disfigured,text,watermark,lowres,",
}
response = requests.post(url=f'{url}/sdapi/v1/txt2img',json=txt2img_data)
response.json
image = Image.open(io.BytesIO(base64.b64decode(response.json()['images'][0])))
image.show()