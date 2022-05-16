FROM python:3
ADD . /
EXPOSE 5000
ENV FLASK_APP=serwer.py
RUN pip install numpy pandas sklearn flask flask_restful
ENTRYPOINT ["flask"]
CMD [ "run", "--host", "0.0.0.0" ]
