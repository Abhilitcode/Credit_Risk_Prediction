FROM python:3.9.12-slim

WORKDIR /creditapp

COPY . /creditapp

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "creditapp.py"]