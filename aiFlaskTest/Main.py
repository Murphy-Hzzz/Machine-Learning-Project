from flask import Flask,render_template,request, jsonify, request, make_response, send_from_directory, abort
import os
from AI_project import app_i2t_pascal,app_t2i_pascal
import  xml.dom.minidom

app = Flask(__name__)
# bootstrap = Bootstrap(app)
basedir = os.path.abspath(os.path.dirname(__file__))

@app.route('/index')
def home():
    return render_template('index.html')


# show photo
@app.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass

@app.route('/text_result', methods=['POST'])
def search_text():
    search_text= request.form.get("message")
    imgsrc,text=app_t2i_pascal.application(testTxt=search_text)
    #imgsrc=["static/photo/1.jpg","static/photo/2.jpg","static/photo/3.jpg","static/photo/4.jpg"]
    # text=""
    # for i in range(4):
    #     # 打开xml文档
    #     try:
    #         dom = xml.dom.minidom.parse(text_path[i])
    #         root = dom.documentElement
    #         cc = dom.getElementsByTagName('text')
    #         c1 = cc[0]
    #         #print (c1.firstChild.data)
    #         text+="第"+str(i)+"段文本： "+c1.firstChild.data
    #     except:
    #         print("xml invalid")
    return render_template('photo_result.html', imgsrc=imgsrc, text=text)

@app.route('/photo_result', methods=['POST'])
def search_photo():
    search_img = request.files.get("photo")
    #print(search_img)
    path = basedir + "/static/photo/"
    file_path = path + search_img.filename
    print(file_path)
    search_img.save(file_path)

    #imgsrc,text=app_t2i.application(testTxt=file_path)
    imgsrc,text_content=app_i2t_pascal.application(testPic=file_path)
    print(file_path)
    #imgsrc=["static/photo/1.jpg","static/photo/2.jpg","static/photo/3.jpg","static/photo/4.jpg"]
    #imgsrc=["static/photo/0c5ec1345e34b51c5b7d6dbc4af7820d.jpg","static/photo/2.jpg","static/photo/3.jpg","static/photo/4.jpg"]
    text=""
    # for i in text_path:
    #     # 打开xml文档
    #     dom = xml.dom.minidom.parse(i)
    #     root = dom.documentElement
    #     cc = dom.getElementsByTagName('text')
    #     c1 = cc[0]
    #     print (c1.firstChild.data)
    #     text+=c1.firstChild.data

    # for i in range(4):
    #     # 打开xml文档
    #     try:
    #         dom = xml.dom.minidom.parse(text_path[i])
    #         root = dom.documentElement
    #         cc = dom.getElementsByTagName('text')
    #         c1 = cc[0]
    #         #print (c1.firstChild.data)
    #         text+="第"+str(i)+"段文本： "+c1.firstChild.data
    #     except:
    #         print("xml invalid")

    #imgsrc[0]="C:/Users/wff/Desktop/AI_proj
    # ect/mnt/data/chenjiefu/linkaiyi/AI_project/data/images/art/0c5ec1345e34b51c5b7d6dbc4af7820d.jpg"
    #imgsrc[0]="AI_project/mnt/data/chenjiefu/linkaiyi/AI_project/data/images/art/17e47ffa2682b0ccb3dc14cea7428436.jpg"
    return render_template('photo_result.html', imgsrc=imgsrc, text=text_content)

if __name__ == '__main__':
    app.run(debug=True)

