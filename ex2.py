import streamlit as st
import random
from sklearn.utils import shuffle
from sklearn.utils.random import check_random_state
from sklearn.utils.validation import check_array
from sklearn.utils.validation import column_or_1d
from sklearn.utils import gen_even_slices
from sklearn.utils import check_consistent_length

appetizers = ['Caesar salad', 'Bruschetta', 'Garlic bread', 'Mozzarella sticks', 'Caprese salad']
entrees = ['Spaghetti and meatballs', 'Grilled salmon', 'Chicken parmesan', 'Shrimp scampi', 'Beef stir fry']
desserts = ['Tiramisu', 'Cheesecake', 'Chocolate cake', 'Creme brulee', 'Ice cream']

class MenuGenerator:
    def __init__(self, items, num_courses=3, random_state=None):
        self.items = items
        self.num_courses = num_courses
        self.random_state = random_state

    def generate_menu(self):
        rng = check_random_state(self.random_state)
        n_samples = len(self.items)
        sample_indices = rng.permutation(n_samples)
        slices = gen_even_slices(self.num_courses, n_samples)
        menu = []

        for slice_ in slices:
            indices = sample_indices[slice_]
            course_items = [self.items[i] for i in indices]
            shuffled_course_items = shuffle(course_items, random_state=rng)
            menu.append(shuffled_course_items[0])

        return menu

menu = []

def generate_menu():
    menu_generator = MenuGenerator(items=[appetizers, entrees, desserts], num_courses=3, random_state=0)
    menu.clear()
    menu.extend(menu_generator.generate_menu())

st.title('Random Menu Generator')

if st.button('Generate Menu'):
    generate_menu()

if menu:
    st.header('Appetizers:')
    st.write(menu[0])
    st.header('Entrees:')
    st.write(menu[1])
    st.header('Desserts:')
    st.write(menu[2])