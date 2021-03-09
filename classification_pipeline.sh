#!/bin/bash
python3 change_config.py -beat initial
python3 ecg_classifier.py
python3 change_config.py -beat mid
python3 ecg_classifier.py
python3 change_config.py -beat final
python3 ecg_classifier.py
