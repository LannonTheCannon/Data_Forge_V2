# def analyze_chart_with_openai(image_path):
#     """
#     Sends the generated chart to OpenAI's GPT-4 Vision API and returns detailed insights.
#     """
#     try:
#         # Convert the image to Base64
#         base64_image = encode_image(image_path)
#
#         # Send request to OpenAI's Vision Model
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": (
#                                 "This is a data visualization chart generated from a dataset related to credit card fraud detection. "
#                                 "Analyze the chart and provide insights by answering the following: \n\n"
#                                 "**1️⃣ Describe the chart:** What type of chart is this (e.g., bar, line, scatter, histogram)? What variables are plotted on the x and y axes?\n\n"
#                                 "**2️⃣ Identify trends & patterns:** Are there any notable trends, peaks, declines, or clusters in the data? Are there any correlations or outliers visible?\n\n"
#                                 "**3️⃣ Interpret key findings:** What does this data suggest in the context of fraud detection? Does it indicate an increase in fraud over time, high fraud in certain categories, or any anomalies?\n\n"
#                                 "**4️⃣ Provide actionable insights:** Based on the data patterns, what conclusions can be drawn, and what recommendations would you suggest for preventing fraud or further investigation?\n\n"
#                                 "Be as detailed as possible and avoid making assumptions beyond what the chart visually represents."
#                             ),
#                         },
#                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
#                     ],
#                 }
#             ],
#             max_tokens=500,
#         )
#
#         return response.choices[0].message["content"]
#
#     except Exception as e:
#         return f"⚠️ Error analyzing chart: {e}"

import openai
import base64
import streamlit as st
from openai import OpenAI

client = openai.OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "temp_chart.png"

# Getting the Base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": '''
                                  "This is a data visualization chart generated from a dataset related to credit card fraud detection. "
                                  "Analyze the chart and provide insights by answering the following: \n\n"
                                  "**1️⃣ Describe the chart:** What type of chart is this (e.g., bar, line, scatter, histogram)? What variables are plotted on the x and y axes?\n\n"
                                  "**2️⃣ Identify trends & patterns:** Are there any notable trends, peaks, declines, or clusters in the data? Are there any correlations or outliers visible?\n\n"
                                  "**3️⃣ Interpret key findings:** What does this data suggest in the context of fraud detection? Does it indicate an increase in fraud over time, high fraud in certain categories, or any anomalies?\n\n"
                                  "**4️⃣ Provide actionable insights:** Based on the data patterns, what conclusions can be drawn, and what recommendations would you suggest for preventing fraud or further investigation?\n\n"
                                  "Be as detailed as possible and avoid making assumptions beyond what the chart visually represents."
                    ''',
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)

print(response.choices[0])