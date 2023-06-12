# Dockerfile

# Base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API and Streamlit app files to the container
COPY . .

EXPOSE 80

# Set the command to run the startup script
CMD ["streamlit","run","app.py","--server.port","80"]