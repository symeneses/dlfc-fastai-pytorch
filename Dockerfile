FROM python:3.8

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs| bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /home
ADD requirements.txt /home
RUN pip install --no-cache-dir -r requirements.txt

ADD requirements-others.txt /home
RUN pip install --no-cache-dir -r requirements-others.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
