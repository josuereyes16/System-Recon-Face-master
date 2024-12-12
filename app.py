from datetime import datetime
from flask import Flask, flash, render_template, request, redirect, url_for, session
import cv2
import base64
import numpy as np
from mtcnn.mtcnn import MTCNN
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------------- Decodificar la imagen --------------------------
def decode_image(image_data):
    encoded_data = image_data.split(',')[1]
    img_data = base64.b64decode(encoded_data)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# ------------------ Función de similitud usando ORB --------------------
def orb_sim(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    similar_regions = [i for i in matches if i.distance < 70]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

# -------------------- Ruta de Login Facial -------------------------
@app.route('/login_face', methods=['POST'])
def login_face():
    if request.method == 'POST':
        image_data = request.form['image']
        frame = decode_image(image_data)
        #face_login = None

        # Detectar el rostro
        detector = MTCNN()
        results = detector.detect_faces(frame)
        if results:
            # odtener face de la imagen
            for result in results:
                x, y, width, height = result['box']  # Obtiene las coordenadas y tamaño del rostro
                x, y = abs(x), abs(y)  # Asegúrate de que las coordenadas sean positivas
                # Extrae el rostro de la imagen
                face_login = frame[y:y + height, x:x + width]
        else:
            flash('No se detectó ningún rostro. Intente de nuevo.')
            return render_template('login.html')

        username = request.form['user_id']
        saved_image_path = f"static/uploads/{username}.jpg"
        saved_face_path = f"static/faces/{username}.jpg"
        
        if os.path.exists(saved_image_path):
            # Comparar rostros usando ORB
            saved_image = cv2.imread(saved_face_path, 0)
            login_image_gray = cv2.cvtColor(face_login, cv2.COLOR_BGR2GRAY)
            similarity = orb_sim(saved_image, login_image_gray)

            if similarity >= 0.95:
                # Obtener la fecha y hora actuales
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Guardar usuario en la sesión
                session['user'] = {'name': username,
                                   'image': saved_image_path,
                                   'last_login': current_time} 
                
                return redirect(url_for('dashboard'))
            else:
                flash('Rostro no coincide. Intente de nuevo.')
                return render_template('login.html')
        else:
            flash('Usuario no encontrado')
            return render_template('login.html')

# ------------------ Ruta de Registro Facial ----------------------
@app.route('/register_face', methods=['POST'])
def register_face():
    if request.method == 'POST':
        image_data = request.form['image']
        frame = decode_image(image_data)
        
        # Detectar el rostro
        detector = MTCNN()
        results = detector.detect_faces(frame)
        if results:
            
            username = request.form['user_id']
            
            # odtener face de la imagen
            for result in results:
                x, y, width, height = result['box']  # Obtiene las coordenadas y tamaño del rostro
                x, y = abs(x), abs(y)  # Asegúrate de que las coordenadas sean positivas
                # Extrae el rostro de la imagen
                face = frame[y:y + height, x:x + width]

                # Guarda el rostro como una nueva imagen
                save_face_path = f"static/faces/{username}.jpg"
                cv2.imwrite(save_face_path, face)
            
            # Guardar imagen de registro
            save_image_path = f"static/uploads/{username}.jpg"
            cv2.imwrite(save_image_path, frame)
            flash('Registro exitoso')
            return redirect(url_for('index'))
        else:
            flash('No se detectó ningún rostro. Intente de nuevo')
            return render_template('register.html')

# ------------------ Ruta del Dashboard --------------------
@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        user_data = session['user']
        return render_template('dashboard.html', user_data=user_data)
    else:
        return redirect(url_for('index'))

# ---------------- Ruta de Inicio --------------------
@app.route('/')
def index():
    return render_template('login.html')

# --------------- Ruta de Registro -------------------
@app.route('/register')
def register():
    return render_template('register.html')

#----------------- Ruta de Salida --------------------
@app.route('/logout')
def logout():
    session.pop('user', None)  # Elimina el usuario de la sesión
    return redirect(url_for('index'))  # Redirige al login después de cerrar sesión

if __name__ == '__main__':
    if not os.path.exists('faces'):
        os.makedirs('faces')
    app.run(debug=True)