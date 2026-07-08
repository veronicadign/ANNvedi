rsync -avz \
    -e "ssh -i ~/.ssh/aws-antelligence-keys.pem" \
    --exclude 'dataset/gooaq-distilroberta-public.hdf5' \
    --exclude '__pycache__/*' \
    --exclude '*.egg-info' \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude 'orthogonal-competition/results.db' \
    . ec2-user@ec2-51-20-42-173.eu-north-1.compute.amazonaws.com:/home/ec2-user/ANNvedi
 