setup:
	pip install -r requirements.txt

pipeline:
	python load_data.py
	python part2_frequency_analysis.py
	python part3_responder_analysis.py
	python part4_subset_analysis.py

dashboard:
	streamlit run dashboard.py --server.headless true
