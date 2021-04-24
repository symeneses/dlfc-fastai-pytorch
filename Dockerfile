FROM python:3.9

WORKDIR /home
ADD requirements.txt /home
RUN pip install --no-cache-dir -r requirements.txt

ADD . /home

EXPOSE 8888

CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
