## Real-Time Face Mask Detection

## Overview:
Created a sophisticated real-time face mask detection system leveraging the power of OpenCV, Tensorflow, Streamlit, and DeepFace. The system not only detects whether a person is wearing a mask but also estimates their age and gender for comprehensive demographic analysis.

## Functionality:
- **Mask Detection:** Utilized the MobileNetV2 pre-trained model for robust mask detection, enabling the system to accurately distinguish between individuals wearing masks and those without.
- **Age and Gender Estimation:** Employed DeepFace models to estimate the age and gender of detected individuals, providing valuable insights for demographic analysis.
- **Real-time Processing:** Leveraged OpenCV for real-time video processing, ensuring seamless detection and analysis of faces in live video streams.
- **User-Friendly Interface:** Deployed the system using Streamlit, offering a user-friendly interface for easy interaction and intuitive visualization of results.

## How it Works:
1. **Face Detection:** Utilized Haar Cascade classifier for efficient face detection in video streams, laying the groundwork for subsequent analysis.
2. **Mask Detection:** Employed the powerful MobileNetV2 model to classify faces as either masked or unmasked, ensuring accurate detection in real-time scenarios.
3. **Age and Gender Prediction:** Leveraged DeepFace models to estimate the age and gender of detected faces, providing additional demographic insights for enhanced analysis.
4. **Streamlit Deployment:** Deployed the complete system using Streamlit, allowing users to seamlessly interact with the application through a web-based interface.

## Use Cases:
- **Retail Analytics:** Deploy the system in supermarkets to analyze the demographics of shoppers and identify age groups more inclined to purchase specific products. This information can inform targeted marketing strategies and product placements.
- **Health Compliance:** Implement the system in public areas where mask-wearing is mandatory, such as airports, hospitals, or public transportation hubs. By automatically detecting individuals without masks, authorities can ensure compliance with health regulations and take necessary actions.
- **Event Management:** Integrate the system into event venues to monitor mask usage and demographics of attendees. Organizers can use this data to enforce safety protocols and optimize event logistics for different age groups.

## Algorithms Used:
- **Face Detection:** Haar Cascade Classifier
- **Mask Detection:** MobileNetV2 (pre-trained on ImageNet)
- **Age and Gender Estimation:** DeepFace Models

## Enhanced Analysis:
By combining mask detection with age and gender estimation, the system offers a holistic approach to understanding mask-wearing behavior within different demographic groups. This comprehensive analysis can inform targeted interventions and public health strategies for mitigating the spread of contagious diseases.