FROM python:3.10
RUN apt update && apt install -y python3-pip
RUN pip3 install numpy pandas scikit-learn seaborn matplotlib --force
WORKDIR /app
COPY grade_prediction_linear_regression.py .
COPY student-mat.csv .
CMD ["python3", "grade_prediction_linear_regression.py" ]