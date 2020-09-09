FROM python:3.7

WORKDIR /home
ADD requirements.txt /home
RUN pip install -r requirements.txt

ADD . /home

EXPOSE 8888

CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
