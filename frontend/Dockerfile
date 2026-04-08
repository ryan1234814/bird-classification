# Use an official Node.js runtime as a parent image
FROM node:20-slim

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy the rest of the frontend code
COPY frontend/ ./

# Expose the port Vite runs on
EXPOSE 5173

# Run the dev server and expose it to the host
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
