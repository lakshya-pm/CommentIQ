from modules.youtube import fetch_video_from_url

url = input("Enter YouTube URL: ")

try:
    data = fetch_video_from_url(url)

    print("\n========== VIDEO DETAILS ==========")
    print("Title       :", data["title"])
    print("Channel     :", data["channel"])
    print("Views       :", data["views"])
    print("Likes       :", data["likes"])
    print("Comments    :", data["comments"])
    print("Published   :", data["published"])
    print("Thumbnail   :", data["thumbnail"])

    print("\nTop 5 Comments:\n")

    for i, comment in enumerate(data["comments_list"][:5], start=1):
        print(f"{i}. {comment}\n")

except Exception as e:
    print("\nError:", e)
