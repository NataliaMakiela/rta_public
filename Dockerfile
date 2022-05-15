FROM python:3
ADD perceptron.py /
EXPOSE 5000:5000
RUN pip install numpy pandas sklearn flask flask_restful
CMD [ "python", "./perceptron.py" ]
