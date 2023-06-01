# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files to the working directory
COPY . .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Set the entry point command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]