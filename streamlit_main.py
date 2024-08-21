import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import tempfile
from PyPDF2 import PdfReader
from openai import OpenAI
import ast 
import os
from langchain_community.utilities import GoogleSerperAPIWrapper

st.set_page_config(layout="wide")
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
search = GoogleSerperAPIWrapper()

def intro():
    st.write("# 欢迎使用文言文助手！📚")
    st.sidebar.success("选择一个功能进行操作。")

    st.markdown(
        """
        文言文助手旨在帮助学生更好地理解和学习文言文，同时为教师提供课程安排的辅助工具。

        ### 功能概览

        - 文言文助手：为学生提供基于文言文的学习和问题解答。
        - 教师助手(施工中)：帮助教师基于提供的材料生成课程大纲和课程内容。

        **👈 从左侧选择一个功能来开始！**
        """
    )

def wenyanwen_assistant():
    @st.cache_resource(show_spinner=False)
    def load_data():
        # 从CSV文件加载数据
        csv_path = "encoded_texts_fixed.csv"
        
        try:
            df = pd.read_csv(csv_path)  # 确保文件名正确
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV文件未找到: {csv_path}")

        # 提取文本
        texts = df['text'].tolist()

        # 提取嵌入向量，使用ast.literal_eval解析字符串为列表
        embeddings = []
        for idx, embedding_str in enumerate(df['embedding']):
            try:
                embedding = ast.literal_eval(embedding_str)
                embeddings.append(embedding)
            except Exception as e:
                print(f"解析嵌入向量时出错：行号 {idx}，内容：{embedding_str}")
                raise ValueError(f"解析嵌入向量时出错: {e}")

        embeddings = np.vstack(embeddings)

        # 初始化Faiss索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return texts, index
    
    texts, index = load_data()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 初始化聊天记录
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "欢迎使用初中文言文助手！请问有什么文言文相关的问题吗？",
            }
        ]

    if prompt := st.chat_input("输入你的文言文问题或段落"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        query_embedding = model.encode([prompt])

        k = 7
        _, indices = index.search(query_embedding, k)
        
        retrieved_texts = [texts[idx] for idx in indices[0]]
        
        #context = "\n".join(retrieved_texts)
        context = search.run(prompt)
        client = OpenAI(api_key = api_key)
        system_setting = """"
            Role（角色）:
            你是一位经验丰富的文言文老师，专门帮助学生解决他们在学习文言文时遇到的各种问题。你只回答与文言文相关的问题，并且需要根据学生提供的资料来给出详细的解答。主动给出更多的范例和提出问题引发学生的思考。

            Profile（个人简介）:
            你通晓中国古代经典文学，精通文言文的词汇、句法和语法结构。你能够准确、清晰地解释文言文中复杂的概念，并帮助学生理解文章的深层含义。你拥有丰富的教学经验，能够针对学生的具体问题给出有针对性的回答。

            Attention（注意事项）:

            只回答与文言文相关的问题，其他学科或领域的问题不予解答。
            回答时应当根据提供的资料，避免脱离上下文或超出资料内容的范围。
            确保回答的内容简洁、明了，并符合学生的理解水平。
            
            Background（背景）:
            学生正在学习文言文，并且需要深入理解课文中的词汇、句式和文化背景。他们可能遇到古今异义词的理解困难、复杂句式的分析挑战，或者需要解释某些特定的语法现象。学生会提供一段相关资料，并基于此提出具体的问题。

            Constraints（约束条件）:

            必须根据学生提供的资料来进行回答，不得超出资料范围。
            解答应当直击学生问题的核心，并提供必要的例证或解释来增强理解。
            每个回答应简洁而深入，避免不必要的冗长或过度复杂的解释。
            
            Goals（目标）:
            帮助学生准确理解文言文中的词汇和句式结构。
            提升学生对文言文整体内容的把握，能够独立分析和解读文言文。
            通过详细的解释和指导，使学生在文言文学习上获得进步和自信。
            
            Skills（技能要求）:
            深入掌握文言文知识，能够在有限的资料范围内进行准确分析。
            出色的解题能力，能够快速定位学生问题中的关键点并提供清晰的解答。
            教学经验丰富，能够根据不同学生的需求调整讲解方式。
            
            Initialization（初始化）:
            当学生提供资料和问题时，你需要首先阅读并理解资料内容，迅速找出与问题相关的部分。然后，针对问题进行详细解答，确保解释清晰且有逻辑性。回答时，请引用资料中的具体内容作为依据，以便学生更好地理解你的解答。
            """
        messages =  [
                {'role':'system',
                'content': system_setting},
                {'role':'user',
                'content': f"相关资料:\n{context}\n\n提问: {prompt}\回答:"},
            ]
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=messages,
            max_tokens=8500,
            temperature=0.0,
        )
        
        st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def teacher_assistant():
    pass



page_names_to_funcs = {
    "学生文言文学习助手": wenyanwen_assistant,
    "介绍": intro,
    "教师助手": teacher_assistant,
}


selected_page = st.sidebar.selectbox("选择页面", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
