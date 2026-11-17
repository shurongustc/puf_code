import gradio as gr,time
from inference import PUFMatcher

matcher=PUFMatcher()

def auth(img):
    if img is None:return None,"No image",0,"",""
    t=time.time();r=matcher.match_topk(img)
    top="\n".join(f"{i+1}. {n}: {s:.4f}" for i,(n,s) in enumerate(r["topk"]))
    return img,r["result"],round(r["similarity"],4),r["matched_id"],top+f"\n\n{(time.time()-t)*1000:.1f} ms"

with gr.Blocks(title="AI-enabled PUF Authentication") as demo:
    gr.Markdown("# AI-enabled Liquid Crystal PUF Authentication\n DINOv2-based physical fingerprint retrieval")

    with gr.Row():
        with gr.Column():
            x=gr.Image(type="pil",label="PUF Response")
            b=gr.Button("Authenticate")
        with gr.Column():
            y=gr.Image(label="Input")
            r=gr.Textbox(label="Result")
            s=gr.Number(label="Similarity")
            i=gr.Textbox(label="Matched ID")
            t=gr.Textbox(label="Top-5 Retrieval")

    b.click(auth,x,[y,r,s,i,t])

demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)