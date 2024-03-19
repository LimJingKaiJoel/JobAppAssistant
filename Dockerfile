# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container at /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Run job_app_asst.py when the container launches
CMD ["python", "job_app_asst.py"]
