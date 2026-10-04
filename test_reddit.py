from modules.reddit import fetch_post_from_url

url = input("Enter Reddit Post URL: ")

try:

    data = fetch_post_from_url(url)

    print("\n========== REDDIT POST ==========\n")

    print("Title       :", data["title"])
    print("Subreddit   :", data["subreddit"])
    print("Author      :", data["author"])
    print("Upvotes     :", data["score"])
    print("Comments    :", data["comments_count"])

    print("\nTop 5 Comments:\n")

    for i, comment in enumerate(data["comments_list"][:5], start=1):

        print(f"{i}. {comment}\n")

except Exception as e:

    print(e)
