#convert
# gcloud auth login
# gcloud auth application-default login

poetry run python ./bucket_conversion.py -max_parts=0 -max_scenes=0
