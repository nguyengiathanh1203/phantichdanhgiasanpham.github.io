import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
import os
import pandas as pd

model = joblib.load("model_random_forest.pkl")
vectorizer = joblib.load("vectorizer_tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def save_prediction(text, label):
    df = pd.DataFrame([[text, label]], columns=["Text", "Prediction"])
    filepath = "predicted_log.csv"
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', index=False, header=False, encoding='utf-8')
    else:
        df.to_csv(filepath, index=False, encoding='utf-8')

def update_history():
    history_listbox.delete(0, tk.END)
    try:
        df = pd.read_csv("predicted_log.csv", names=["Text","Prediction"], skiprows=1)
        for _, row in df.tail(10).iterrows():
            snippet = row["Text"][:50].replace("\n"," ") + "..."
            history_listbox.insert(tk.END, f"[{row['Prediction']}] {snippet}")
    except FileNotFoundError:
        pass


def update_stats():
    try:
        df = pd.read_csv("predicted_log.csv", names=["Text","Prediction"], skiprows=1)
        counts = df["Prediction"].value_counts()
        stats.set(
            f"Tích cực: {counts.get('positive',0)} | "
            f"Trung lập: {counts.get('neutral',0)} | "
            f"Tiêu cực: {counts.get('negative',0)}"
        )
    except FileNotFoundError:
        stats.set("Chưa có dữ liệu dự đoán")


def predict_text(input_text):
    vec = vectorizer.transform([input_text])
    pred = model.predict(vec)[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label


def on_predict():
    txt = text_input.get("1.0", tk.END).strip()
    if not txt:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập hoặc chọn văn bản.")
        return

    lines = txt.split('\n')
    results = []
    for line in lines:
        if line.strip():
            label = predict_text(line)
            results.append(f"{label.upper()} - {line.strip()}")
            save_prediction(line.strip(), label)

    result_var.set(f"{len(results)} đánh giá đã được dự đoán.")
    update_history()
    update_stats()


def on_clear():
    text_input.delete("1.0", tk.END)
    result_var.set("")


def on_load_file():
    path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not path:
        return

    content = ""
    if path.endswith(".csv"):
        try:
            df = pd.read_csv(path)
            if "Text" in df.columns:
                content = "\n".join(str(t) for t in df["Text"].dropna())
            else:
                messagebox.showerror("Lỗi", "File CSV phải có cột 'Text'.")
                return
        except Exception as e:
            messagebox.showerror("Lỗi đọc file CSV", str(e))
            return
    else: 
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

    text_input.delete("1.0", tk.END)
    text_input.insert(tk.END, content)


root = tk.Tk()
root.title("Sentiment Analysis GUI")
root.geometry("720x670")


tk.Label(root, text="Nhập nội dung hoặc chọn file:", font=("Arial", 12)).pack(pady=5)
text_input = tk.Text(root, height=8, width=85)
text_input.pack(pady=5)


btn_frame = tk.Frame(root)
tk.Button(btn_frame, text="📂 Chọn file", command=on_load_file).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="▶️ Dự đoán", command=on_predict, bg="lightblue").pack(side=tk.LEFT)
tk.Button(btn_frame, text="🧹 Xóa đánh giá", command=on_clear, bg="lightgray").pack(side=tk.LEFT, padx=10)
btn_frame.pack(pady=10)


result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, font=("Arial", 14, "bold"), fg="blue").pack(pady=5)


stats = tk.StringVar()
tk.Label(root, textvariable=stats, font=("Arial", 12), fg="green").pack(pady=5)


tk.Label(root, text="🧾 Lịch sử dự đoán gần nhất:", font=("Arial", 11)).pack(pady=5)
history_listbox = tk.Listbox(root, width=100, height=12)
history_listbox.pack(pady=5)


update_history()
update_stats()

root.mainloop()

