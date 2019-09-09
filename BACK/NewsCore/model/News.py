from datetime import datetime
import uuid

def current_time():
    return datetime.now() #.strftime('%Y-%m-%d %H:%M:%S')

class News:

    def __init__(self, news_number, typology, punctuation, title, summary, graph, side, metrics):
        self.__id = uuid.uuid1()
        self.__date = current_time()
        self.__news_number = news_number
        self.__typology = typology
        self.__punctuation = punctuation
        self.__title = title
        self.__summary = summary
        self.__graph = graph
        self.__side = side
        self.__metrics = metrics



    def get_id(self):
        return self.__id

    def get_date(self):
        return self.__date

    def get_news_number(self):
        return self.__news_number

    def get_typology(self):
        return self.__typology

    def get_punctuation(self):
        return self.__punctuation

    def get_title(self):
        return self.__title

    def get_summary(self):
        return self.__summary

    def get_graph(self):
        return self.__graph

    def get_side(self):
        return self.__side

    def get_metrics(self):
        return self.__metrics



    def set_id(self, id):
        self.__id = id

    def set_date(self, date):
        self.__date = date

    def set_news_number(self, news_number):
        self.__news_number = news_number

    def set_typology(self, typology):
        self.__typology = typology

    def set_punctuation(self, punctuation):
        self.__punctuation = punctuation

    def set_title(self, title):
        self.__title = title

    def set_summary(self, summary):
        self.__summary = summary

    def set_graph(self, graph):
        self.__graph = graph

    def set_side(self, side):
        self.__side = side

    def set_metrics(self, metrics):
        self.__metrics = metrics




# if __name__ == '__main__':
#     noticia1 = News('PUE', 'hi', 'ho')
#     print(noticia1.get_summary())
#     noticia1.set_summary('hola')
#     print(noticia1.get_summary())
