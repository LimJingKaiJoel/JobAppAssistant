# Use an official Python runtime as a parent image
FROM python:3.11

#WORKDIR /main
COPY . .
COPY static/ static/
COPY InternData.csv /main
COPY EntryData.csv /main
COPY SeniorData.csv /main
#WORKDIR /main


# Copy the current directory contents into the container at /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade --force-reinstall pymupdf


EXPOSE 5000

# Run job_app_asst.py when the container launches
CMD ["python", "main/app.py"]
#CMD tail -f /dev/null
