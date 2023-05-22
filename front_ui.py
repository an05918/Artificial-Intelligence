from enum import auto
from re import MULTILINE
from typing import List
from kivy.core import text
from kivy.uix.gridlayout import GridLayout
import pandas as pd
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage

from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty
from kivy.uix.recyclegridlayout import RecycleGridLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.scrollview import ScrollView

from BooksRecommendationSystem import*


class MainWindow(Screen):
    def on_kv_post(self, *args):            #overall layout of calculator's structure and buttons
        main_layout = BoxLayout(orientation="vertical")

        # input bar with button
        recommend_sec_input = BoxLayout(orientation="horizontal", size_hint=(1, 0.2))
        recommend_sec_label = BoxLayout(orientation="horizontal",size_hint=(1, 0.2))

        # all of the labels (book name, number of recommendations, and submit)
        label_book_inp = Label(text="Enter book name", size_hint=(0.6, 1))
        label_num_out = Label(text="Num of Recommendations", size_hint=(0.2, 1))
        label_empty = Label(text="", size_hint=(0.2, 1))

        # adding the labels to the boxes one after the other
        recommend_sec_label.add_widget(label_book_inp)
        recommend_sec_label.add_widget(label_num_out)
        recommend_sec_label.add_widget(label_empty)


        # this has 3 inputs, the book name textbox, a dropdown for desired number of results, and a recommend button
        self.booknameinp = TextInput(multiline=False, readonly=False, halign="left", font_size=17, size_hint=(0.6,1))
        self.num_out_drop = self.time_2_spinner = Spinner(text='1', values=(
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '50', '100'), background_color = [128, 128, 128, 0.2], size_hint=(0.2,1))
        recommend_btn = Button(text="Recommend", size_hint=(0.2, 1), background_color = [128, 128, 128, 0.2])
        recommend_btn.bind(on_press = self.recommend_selected)


        # adding the inputs to the boxes one after the other
        recommend_sec_input.add_widget(self.booknameinp)
        recommend_sec_input.add_widget(self.num_out_drop)
        recommend_sec_input.add_widget(recommend_btn)

        # adding the sections one by one vertically (top to bottom)
        main_layout.add_widget(recommend_sec_label)
        main_layout.add_widget(recommend_sec_input)


        # book1_sec_above = ScrollView(size_hint=(1, None), size=(Window.width, Window.height))
        # book1_sec_above = BoxLayout(orientation = "vertical")
        # book1_sec = BoxLayout(orientation="horizontal")
        # # book2_sec = BoxLayout(orientation="horizontal")
        # # book3_sec = BoxLayout(orientation="horizontal")
        # book1_text = Label(text="Harry potter", size_hint=(0.6, 1))
        # book1 = AsyncImage(source='http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg', size_hint=(0.4,1))
        # book2_text = Label(text="Harry potter", size_hint=(0.6, 1))
        # book2 = AsyncImage(source='http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg', size_hint=(0.4,1))
        # book3_text = Label(text="Harry potter", size_hint=(0.6, 1))
        # book3 = AsyncImage(source='http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg', size_hint=(0.4,1))

        # book1_sec.add_widget(book1_text)
        # book1_sec.add_widget(book1)
        # # book2_sec.add_widget(book2_text)
        # # book2_sec.add_widget(book2)
        # # book3_sec.add_widget(book3_text)
        # # book3_sec.add_widget(book3)
        

        # book1_sec_above.add_widget(book1_sec)
        # # book1_sec_above.add_widget(book2_sec)
        # # book1_sec_above.add_widget(book3_sec)
        # # book1_sec_above.add_widget(book1_sec)
        # # book1_sec_above.add_widget(book1_sec)

        # main_layout.add_widget(book1_sec_above)





        self.ids.b_layout.add_widget(main_layout)

    def recommend_selected(self, instance):
        recommendedBooks = []
        # recommendedBooks = RecommendationSystem(self.booknameinp.text, self.num_out_drop.text)

        # recommendedBooks = [["Harry Potter and the Prisoner of Azkaban (Book 3)", "Harry Potter and the Goblet of Fire (Book 4)", "Harry Potter and the Order of the Phoenix (Book 5)","Harry Potter and the Chamber of Secrets (Book 2)", "Fried Green Tomatoes at the Whistle Stop Cafe"], ["Harry Potter and the Sorcerer's Stone (Book 1)", "Harry Potter and the Goblet of Fire (Book 4)", "Harry Potter and the Chamber of Secrets (Book 2)", "Harry Potter and the Prisoner of Azkaban (Book 3)", "Harry Potter and the Order of the Phoenix (Book 5)"]]
        
        # recommendedBooks = [[('Harry Potter and the Prisoner of Azkaban (Book 3)', 'http://images.amazon.com/images/P/0439136369.01.MZZZZZZZ.jpg'), ('Harry Potter and the Goblet of Fire (Book 4)', 'http://images.amazon.com/images/P/0439139597.01.MZZZZZZZ.jpg'), ('Harry Potter and the Order of the Phoenix (Book 5)', 'http://images.amazon.com/images/P/043935806X.01.MZZZZZZZ.jpg'), ('Harry Potter and the Chamber of Secrets (Book 2)', 'http://images.amazon.com/images/P/0439064864.01.MZZZZZZZ.jpg'), ('Fried Green Tomatoes at the Whistle Stop Cafe', 'http://images.amazon.com/images/P/0804115613.01.MZZZZZZZ.jpg')], [("Harry Potter and the Sorcerer's Stone (Book 1)", 'http://images.amazon.com/images/P/043936213X.01.MZZZZZZZ.jpg'), ("Harry Potter and the Sorcerer's Stone (Book 1)", 'http://images.amazon.com/images/P/0590353403.01.MZZZZZZZ.jpg'), ("Harry Potter and the Sorcerer's Stone (Book 1)", 'http://images.amazon.com/images/P/0590353403.01.MZZZZZZZ.jpg'), ("Harry Potter and the Sorcerer's Stone (Book 1)", 'http://images.amazon.com/images/P/043936213X.01.MZZZZZZZ.jpg'), ("Harry Potter and the Sorcerer's Stone (Book 1)", 'http://images.amazon.com/images/P/0590353403.01.MZZZZZZZ.jpg')]]
        # recommendedBooks = [[("Harry Potter and the Prisoner of Azkaban (Book 3)","http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg"), ("Harry Potter and the Goblet of Fire (Book 4)", "http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg"), ("Harry Potter and the Order of the Phoenix (Book 5)", "http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg"),("Harry Potter and the Chamber of Secrets (Book 2)","http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg")], [("Harry Potter and the Sorcerer's Stone (Book 1)","http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg"), ("Harry Potter and the Goblet of Fire (Book 4)","http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg"), ("Harry Potter and the Chamber of Secrets (Book 2)","http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg"), ("Harry Potter and the Prisoner of Azkaban (Book 3)","http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg")]]
        recommendedBooks = [[('Harry Potter and the Prisoner of Azkaban (Book 3)', 'http://images.amazon.com/images/P/0439136369.01.MZZZZZZZ.jpg'), ('Harry Potter and the Goblet of Fire (Book 4)', 'http://images.amazon.com/images/P/0439139597.01.MZZZZZZZ.jpg'), ('Harry Potter and the Order of the Phoenix (Book 5)', 'http://images.amazon.com/images/P/043935806X.01.MZZZZZZZ.jpg'), ('Harry Potter and the Chamber of Secrets (Book 2)', 'http://images.amazon.com/images/P/0439064864.01.MZZZZZZZ.jpg'), ('Fried Green Tomatoes at the Whistle Stop Cafe', 'http://images.amazon.com/images/P/0804115613.01.MZZZZZZZ.jpg')], [("Harry Potter and the Sorcerer's Stone (Book 1)", 'http://images.amazon.com/images/P/043936213X.01.MZZZZZZZ.jpg'), ('Harry Potter and the Goblet of Fire (Book 4)', 'http://images.amazon.com/images/P/0439139600.01.MZZZZZZZ.jpg'), ('Harry Potter and the Chamber of Secrets (Book 2)', 'http://images.amazon.com/images/P/0439064872.01.MZZZZZZZ.jpg'), ('Harry Potter and the Prisoner of Azkaban (Book 3)', 'http://images.amazon.com/images/P/0439136369.01.MZZZZZZZ.jpg'), ('Harry Potter and the Order of the Phoenix (Book 5)', 'http://images.amazon.com/images/P/043935806X.01.MZZZZZZZ.jpg')]]

        collaborative_label = Label(text="Recommended Books via Collaborative Filtering", size_hint=(1, 1), color = [140, 254, 57, 1])
        self.ids.b_layout.add_widget(collaborative_label)

        for cell in recommendedBooks[0]:
            book_row = BoxLayout(orientation="horizontal")
            book_name_label = Label(text=cell[0], size_hint=(1, 1))
            book_image = AsyncImage(source=cell[1], size_hint=(1,1))

            book_row.add_widget(book_name_label)
            book_row.add_widget(book_image)
            self.ids.b_layout.add_widget(book_row)


        content_based_label = Label(text="Recommended Books via Content Based Filtering", size_hint=(1, 1), color = [140, 254, 57, 1])
        self.ids.b_layout.add_widget(content_based_label)

        for cell in recommendedBooks[1]:
            book_row = BoxLayout(orientation="horizontal")
            book_name_label = Label(text=cell[0], size_hint=(1, 1))
            book_image = AsyncImage(source=cell[1], size_hint=(1,1))

            book_row.add_widget(book_name_label)
            book_row.add_widget(book_image)
            self.ids.b_layout.add_widget(book_row)

        # data_set = [{'text': str(x[0]), 'Index': str(x[1]), 'selectable': True} for x in cell_rv]
        # self.ids.rv.data = data_set





class WindowManager(ScreenManager):
    pass


kivy_file = Builder.load_file("design_b_end.kv")   #The Kivy design file is being loaded



class BookRecommendationSys(App):
    def build(self):
        return kivy_file


if __name__ == "__main__":
    app = BookRecommendationSys()
    app.run()
