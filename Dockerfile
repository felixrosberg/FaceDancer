FROM nvcr.io/nvidia/pytorch:21.03-py3

WORKDIR /workspace
COPY . . 

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 tzdata -y

RUN pip install fastapi
RUN pip install python-multipart
RUN pip install uvicorn
RUN pip install tensorflow-addons
RUN pip install scipy
RUN pip install scikit-image
RUN pip install moviepy==1.0.3
RUN pip install imageio
RUN pip install imageio-ffmpeg
RUN pip install requests
RUN pip install tqdm
RUN pip install tensorflow
RUN pip install opencv-python==4.5.5.64

ENTRYPOINT ["python", "api.py"]
