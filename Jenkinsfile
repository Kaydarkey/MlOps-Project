pipeline {
    agent any

    environment {
        IMAGE_NAME = "epl-inference-api"
        APP_PATH = "inference"
        DOCKERFILE_PATH = "inference/Dockerfile"
        COMPOSE_FILE = "docker-compose.yml"
        // Uncomment and set these if pushing to ECR
        // AWS_REGION = credentials('aws-region')
        // AWS_ACCOUNT_ID = credentials('aws-account-id')
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source code...'
                checkout scm
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Installing Python dependencies for local testing...'
                sh 'pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run MLOps Pipeline') {
            steps {
                echo 'Running the full data and training pipeline via Prefect...'
                sh 'python pipelines/train_model_dag.py'
            }
        }

        stage('Build Docker Images') {
            steps {
                echo 'Building all Docker images with docker-compose...'
                sh 'docker-compose build'
            }
        }

        stage('Start Services') {
            steps {
                echo 'Starting all services with docker-compose...'
                sh 'docker-compose up -d'
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running tests inside the inference_api container...'
                sh 'docker-compose run --rm inference_api pytest tests/'
            }
        }

        // Optionally, add a test stage here to run integration tests

        stage('Stop Services') {
            steps {
                echo 'Stopping all services...'
                sh 'docker-compose down'
            }
        }

        // stage('Push API Image to ECR') {
        //     environment {
        //         AWS_REGION = credentials('aws-region')
        //         AWS_ACCOUNT_ID = credentials('aws-account-id')
        //     }
        //     steps {
        //         withCredentials([
        //             string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
        //             string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
        //         ]) {
        //             sh '''
        //             echo "Logging in to AWS ECR..."
        //             aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
                    
        //             echo "Tagging Docker image..."
        //             docker tag $IMAGE_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$IMAGE_NAME:latest
                    
        //             echo "Pushing Docker image to ECR..."
        //             docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$IMAGE_NAME:latest
        //             '''
        //         }
        //     }
        // }
    }

    post {
        always {
            echo 'Pipeline finished.'
            // Clean up workspace or send notifications here
        }
    }
} 