import streamlit as st
import requests

st.title("Git Hub User Lookup")
username=st.text_input("Enter a Github username","octocat")
if st.button("Look up"):
    response = requests.get(f"https://api.github.com/users/{username}",timeout=5)
    if response.status_code == 200:
        data = response.json()
        st.write(f"Name: {data['name']}")
        st.write(f"Public repos:{data['public_repos']}")
        st.image(data["avatar_url"],width=100)
    else:
        st.write(f"User not found (status{response.status_code})")
        
