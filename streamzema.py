import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from skimage.measure import regionprops_table
from skimage import io, color, measure, img_as_ubyte, feature, filters, exposure

# Load the image and process it once, making `img` available for all menu options
image_path = r"C:\Users\ASUS\Downloads\Eczema_Project\eczema-subacute-35-NoWM.jpeg.jpg"
im = io.imread(image_path)
img = color.rgb2gray(im)  # Convert image to grayscale
img = img_as_ubyte(img)   # Convert to uint8

# Sidebar menu
with st.sidebar:
    selected = option_menu("Pengolahan Citra Medika", ["Home", "Encyclopedia", "Feature Extraction", "Chatbot"], default_index=0)

# Home Page
if selected == "Home":
    st.title('Final Project ✨')
    st.subheader("Anggota kelompok")
    group_members = [
        "Farhan Majid - 5023211049",
        "Leony Purba - 50232110",
        "Benedicta Sabdaningtyas - 50232110",
        "Adelia Safira - 50232110"
    ]
    for member in group_members:
        st.markdown(f"<p style='font-family:Georgia; color: black; font-size: 20px;'>{member}</p>", unsafe_allow_html=True)

# Eczema Subacute Page
elif selected == "Encyclopedia":
    st.markdown("<h1 style='text-align: center; color: red;'>🫀ENCYCLOPEDIA</h1>", unsafe_allow_html=True)
    questions = [
        ("1. Apa itu Eczema Subacute?", "Eczema subacute adalah bentuk eksim yang ditandai dengan gejala peradangan kulit yang sedang berlangsung, muncul setelah fase akut eksim. Pada tahap ini, gejala yang umum terjadi meliputi rasa gatal yang mengganggu, kemerahan dan pembengkakan pada kulit, kekeringan dan pengelupasan, serta lesi yang dapat mengeluarkan cairan atau membentuk keropeng saat mengering."),
        ("2. Apa gejala umum yang terjadi pada eczema subacute?", "Eczema subacute dapat dipicu oleh berbagai faktor, seperti alergen (serbuk sari atau debu), iritan (deterjen atau sabun), perubahan cuaca yang ekstrem, dan stres emosional. Pengobatan untuk eczema subacute biasanya melibatkan penggunaan krim atau salep kortikosteroid untuk mengurangi peradangan dan gatal, serta pelembap untuk menjaga kelembapan kulit. Penting juga untuk mengidentifikasi dan menghindari alergen atau iritan yang dapat memperburuk kondisi. Jika tidak ditangani dengan baik, eczema subacute dapat berkembang menjadi eczema kronis, sehingga mencari perawatan yang tepat dan mengikuti saran dokter sangatlah penting untuk mengelola gejala dan mencegah kekambuhan."),
    ]
    for title, description in questions:
        st.markdown(f"<p style='font-family:Georgia; color:yellow; font-size: 23px; text-align: left;'>{title}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-family:Georgia; color:white; font-size: 20px; text-align: justify;'>{description}</p>", unsafe_allow_html=True)
    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/fmurdUlmaIg" 
                frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen></iframe>""", unsafe_allow_html=True)

# Eczema Feature Extraction Page
elif selected == "Feature Extraction":
    st.title("Eczema Image Feature Extraction 🧬")
    selected2 = option_menu(
        None, 
        ["Image", "Image Processing", "Edge Detection", "Image Segmentation", "Data"], 
        icons=['image', 'adjust', 'filter', 'table'], 
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal"
    )

    # Image Section
    if selected2 == "Image":
        st.subheader("Image Section")
        st.image(im, caption="Loaded Eczema Image", use_column_width=True)
        st.subheader("Grayscale Image (Converted to uint8)")
        st.image(img, caption="Grayscale Image", use_column_width=True)

    # Image Processing Section
    elif selected2 == "Image Processing":
        st.subheader("Pre-Processing")
        
        # Calculate Otsu's Threshold
        threshold = filters.threshold_otsu(img)
        st.write(f"Otsu's threshold value: {threshold}")

        # Apply Adaptive Histogram Equalization (AHE)
        img_hieq = exposure.equalize_adapthist(img, clip_limit=0.9) * 255  
        img_hieq = img_hieq.astype('uint8')

        # Display original image with Otsu's contour and binary thresholded image
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        ax[0].imshow(img, cmap='gray')
        ax[0].contour(img, levels=[threshold], colors='red')
        ax[0].set_title('Original Image with Otsu Contour')
        ax[1].imshow(img < threshold, cmap='gray')
        ax[1].set_title('Binary Image (Otsu Threshold Applied)')
        st.pyplot(fig)

        # Display AHE result
        st.subheader("Adaptive Histogram Equalization")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_hieq, cmap='gray')
        ax.set_title('Adaptive Histogram Equalization')
        st.pyplot(fig)

        # Otsu Thresholding on AHE result
        st.subheader("Otsu Thresholding on AHE Result")
        binary_image = img_hieq < filters.threshold_otsu(img_hieq)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(binary_image, cmap='gray')
        ax.set_title('Binary Image (Otsu Threshold on AHE)')
        st.pyplot(fig)

        # Remove small objects
        st.subheader("Remove Small Objects")
        from skimage import morphology
        only_large_blobs = morphology.remove_small_objects(binary_image, min_size=100)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(only_large_blobs, cmap='gray')
        ax.set_title('Binary Image with Small Objects Removed')
        st.pyplot(fig)

        # Fill small holes
        st.subheader("Fill Small Holes")
        only_large = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs), min_size=100))
        image_segmented = only_large  # Save result for later use

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_segmented, cmap='gray')
        ax.set_title('Binary Image with Small Holes Filled')
        st.pyplot(fig)

    # Edge Detection Section
    elif selected2 == "Edge Detection":
        st.subheader("Edge Detection Filters")
        
        # Re-apply segmentation if needed
        from skimage import morphology
        img_hieq = exposure.equalize_adapthist(img, clip_limit=0.9) * 255
        binary_image = img_hieq < filters.threshold_otsu(img_hieq)
        only_large_blobs = morphology.remove_small_objects(binary_image, min_size=100)
        only_large = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs), min_size=100))
        image_segmented = only_large

        # Apply edge detection filters
        roberts = filters.roberts(image_segmented)
        sobel = filters.sobel(image_segmented)
        prewitt = filters.prewitt(image_segmented)
        canny = feature.canny(image_segmented, sigma=1)

        # Display edge detection results in a 2x2 grid
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].imshow(roberts, cmap='gray')
        ax[0, 0].set_title('Roberts')
        ax[0, 1].imshow(sobel, cmap='gray')
        ax[0, 1].set_title('Sobel')
        ax[1, 0].imshow(prewitt, cmap='gray')
        ax[1, 0].set_title('Prewitt')
        ax[1, 1].imshow(canny, cmap='gray')
        ax[1, 1].set_title(r'Canny $\sigma=1$')

        for a in ax.flat:
            a.axis('off')

        st.pyplot(fig)

    # Bagian dalam blok "Image Segmentation"
    elif selected2 == "Image Segmentation":
        st.subheader("Contour Image")
        from skimage import morphology
        import math
        from skimage.measure import label, regionprops
        
        img_hieq = exposure.equalize_adapthist(img, clip_limit=0.9) * 255
        binary_image = img_hieq < filters.threshold_otsu(img_hieq)
        only_large_blobs = morphology.remove_small_objects(binary_image, min_size=100)
        only_large = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs), min_size=100))
        image_segmented = only_large
        
        # Periksa apakah `image_segmented` dan `img` valid
        if 'image_segmented' in locals() and 'img' in locals():
            # Konversi tipe data ke uint8
            image_segmented = img_as_ubyte(image_segmented)
            threshold = filters.threshold_otsu(img) 
            
            # Menampilkan gambar dengan kontur
            fig, ax = plt.subplots()
            ax.imshow(image_segmented, cmap='gray')
            ax.contour(image_segmented, [threshold])
            
            # Menampilkan plot di Streamlit
            st.pyplot(fig)

    # Tambahan kode untuk pewarnaan acak pada label
        from matplotlib.colors import ListedColormap
        from scipy import ndimage as ndi

        lab_image = image_segmented
        rand_cmap = ListedColormap(np.random.rand(256, 3))  # Membuat colormap acak
        labels, nlabels = ndi.label(lab_image)

        labels_for_display = np.where(labels > 0, labels, np.nan)
        
        # Menampilkan gambar dengan label acak
        fig2, ax2 = plt.subplots()
        ax2.imshow(lab_image, cmap='gray')
        ax2.imshow(labels_for_display, cmap=rand_cmap)
        ax2.axis('off')
        ax2.set_title(f'Ezcema Subacute Labeled ({nlabels} labels)')
        st.pyplot(fig2)
        
        # Melakukan labeling pada objek yang ditemukan
        boxes = ndi.find_objects(labels)
        for label_ind, label_coords in enumerate(boxes):
            if label_coords is None:
                continue  # Jika label tidak valid, lewati

            cell = lab_image[label_coords]
            
            # Filter objek berdasarkan ukuran
            cell_size = np.prod(cell.shape)

            if cell_size < 5000: 
                lab_image = np.where(labels == label_ind + 1, 0, lab_image)
        
        # Regenerasi label setelah filter
        labels, nlabels = ndi.label(lab_image)
        st.write(f'Terdapat {nlabels} komponen / objek yang terdeteksi setelah filtering.')

        # Menampilkan subset dari objek yang terdeteksi
        fig3, axes = plt.subplots(nrows=1, ncols=6, figsize=(10, 6))
        for ii, obj_indices in enumerate(ndi.find_objects(labels)[5:11]):
            if obj_indices is not None:
                cell = image_segmented[obj_indices]
                axes[ii].imshow(cell, cmap='gray')
                axes[ii].axis('off')
                axes[ii].set_title(f'Label #{ii+1}\nUkuran: {cell.shape}')
        
        plt.tight_layout()
        st.pyplot(fig3)
    
    # Menjalankan labeling dan menganalisis properti region

        label_img = label(lab_image)
        regions = regionprops(label_img)
        
        # Menampilkan centroid dan orientasi pada gambar
        fig3, ax3 = plt.subplots()
        ax3.imshow(lab_image, cmap=plt.cm.gray)
        
        for props in regions:
            y0, x0 = props.centroid
            orientation = props.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

            # Plot centroid, orientasi, dan bounding box
            ax3.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            ax3.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            ax3.plot(x0, y0, '.g', markersize=15)

            # Plot bounding box
            minr, minc, maxr, maxc = props.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax3.plot(bx, by, '-b', linewidth=2.5)

        ax3.set_title("Centroid and Orientation of Labeled Regions")
        st.pyplot(fig3)
        
    # Data Extraction Section
    elif selected2 == "Data":
        st.subheader("Extracted Data")
        # Path ke file Excel
    excel_path = r"C:\Users\50232\Downloads\extract_features.xlsx"

    # Data baru yang ingin ditulis ke file Excel
    df_new = pd.DataFrame({
        'Column1': [1, 2, 3],
        'Column2': ['A', 'B', 'C']
    })

    # Menulis data ke file Excel tanpa memeriksa keberadaan file
    # Mode 'w' akan membuat file baru jika belum ada atau menggantikan file yang sudah ada
    with pd.ExcelWriter(excel_path, mode='w', engine="openpyxl") as writer:
        df_new.to_excel(writer, index=False, sheet_name='New Features')

    st.write("Data berhasil ditambahkan ke file Excel.")


# Eczema Chatbot Page
elif selected == "Chatbot":
    st.title("Eczema Chatbot 🤖")

    # Initialize chat history if not already in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Predefined questions and answers for the chatbot
    qa_pairs = {
        "What is eczema?": "Eczema is a condition that makes your skin red and itchy. It's common in children but can occur at any age.",
        "What are the symptoms of eczema?": "Symptoms include red to brownish-gray patches, itching (especially at night), and small, raised bumps that may leak fluid.",
        "How can I prevent eczema flare-ups?": "To prevent flare-ups, keep your skin moisturized, avoid triggers (like certain soaps, allergens, and stress), and avoid scratching.",
        "What treatments are available for eczema?": "Treatments may include moisturizers, steroid creams, and other topical or oral medications prescribed by a doctor.",
        "Can diet affect eczema?": "Certain foods may trigger eczema for some people, such as dairy, eggs, or nuts. It's best to consult a healthcare provider if you suspect food triggers.",
        "Is eczema contagious?": "No, eczema is not contagious. It’s a chronic condition often related to genetics and environmental triggers.",
    }

    # Display predefined questions for user to choose
    st.write("### Ask me about Eczema:")
    user_question = st.selectbox("Choose a question:", [""] + list(qa_pairs.keys()))

    # Display the answer if a question is selected
    if user_question:
        answer = qa_pairs.get(user_question)
        st.write(f"**Bot:** {answer}")
