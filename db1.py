class Chat:
    """Class for saving the context to the Chroma db while chatting with GPT"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        # if you need a memory "on the go"
        # self.memory = ConversationBufferMemory(memory_key='chat_history')
        self.prompt = PromptTemplate(input_variables=["chat_history", "human_input"],
                                     template=TEMPLATE)

        self.llm_chain = LLMChain(
            llm=OpenAI(),
            prompt=self.prompt,
            verbose=True
        )
        # if you want to save memory locally via the Chroma
        self.vectordb = self._get_vector()

    def _get_vector(self):
        """Instruction for finding the Chroma db locally or create the new one."""
        if os.path.isdir('db'):
            vectordb = Chroma(persist_directory='db',
                              embedding_function=self.embeddings)
        else:
            text_splitter = RecursiveCharacterTextSplitter()
            start_chunk = text_splitter.split_text("You are a helpful assistant")
            vectordb = Chroma.from_texts(start_chunk,
                                         self.embeddings,
                                         persist_directory='db')
            vectordb.persist()
        return vectordb
