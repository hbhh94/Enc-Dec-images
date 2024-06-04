import numpy as np
from PIL import Image

def calculate_mse(original_image, decrypted_image):
    mse = np.mean((original_image - decrypted_image) ** 2)
    return mse

def encryp_8dhyperchaos(I, key):
    if len(I.shape) == 3:
        q, t, c = I.shape  # the size of image and its channels
    else:
        q, t = I.shape  # the size of image
        c = 1  # set default channels to 1 for grayscale images
    N = 1 + q * t * c
    h = 0.01
    # generate 8 group random number
    np.random.seed(key)
    x1 = np.random.rand(N)
    x2 = np.random.rand(N)
    x3 = np.random.rand(N)
    x4 = np.random.rand(N)
    x5 = np.random.rand(N)
    x6 = np.random.rand(N)
    x7 = np.random.rand(N)
    x8 = np.random.rand(N)

    for n in range(N-1):
        k1 = 10 * (x2[n] - x1[n]) + x4[n]
        j1 = 76 * x1[n] - x1[n] * x3[n] + x4[n]
        r1 = x1[n] * x2[n] - x3[n] - x4[n] + x7[n]
        t1 = -3 * (x1[n] + x2[n]) + x5[n]
        a1 = -x2[n] - 0.2 * x4[n] + x6[n]
        b1 = (-0.1) * (x1[n] + x5[n]) + 0.2 * x7[n]
        c1 = (-0.1) * (x1[n] + x6[n] - x8[n])
        d1 = -0.2 * x7[n]

        k2 = 10 * (x2[n] + j1 * h / 2 - x1[n] - k1 * h / 2) + x4[n] + t1 * h / 2
        j2 = 76 * (x1[n] + k1 * h / 2) - (x1[n] + k1 * h / 2) * (x3[n] + r1 * h / 2) + x4[n] + t1 * h / 2
        r2 = (x1[n] + k1 * h / 2) * (x2[n] + j1 * h / 2) - x3[n] - r1 * h / 2 - x4[n] - t1 * h / 2 + x7[n] + c1 * h / 2
        t2 = -3 * (x1[n] + k1 * h / 2 + x2[n] + j1 * h / 2) + x5[n] + a1 * h / 2
        a2 = -(x2[n] + j1 * h / 2) - 0.2 * (x4[n] + t1 * h / 2) + x6[n] + b1 * h / 2
        b2 = (-0.1) * (x1[n] + k1 * h / 2 + x5[n] + a1 * h / 2) + 0.2 * (x7[n] + c1 * h / 2)
        c2 = (-0.1) * (x1[n] + k1 * h / 2 + x6[n] + b1 * h / 2 - x8[n] - d1 * h / 2)
        d2 = -0.2 * (x7[n] + c1 * h / 2)

        k3 = 10 * (x2[n] + j2 * h / 2 - x1[n] - k2 * h / 2) + x4[n] + t2 * h / 2
        j3 = 76 * (x1[n] + k2 * h / 2) - (x1[n] + k2 * h / 2) * (x3[n] + r2 * h / 2) + x4[n] + t2 * h / 2
        r3 = (x1[n] + k2 * h / 2) * (x2[n] + j2 * h / 2) - x3[n] - r2 * h / 2 - x4[n] - t2 * h / 2 + x7[n] + c2 * h / 2
        t3 = -3 * (x1[n] + k2 * h / 2 + x2[n] + j2 * h / 2) + x5[n] + a2 * h / 2
        a3 = -(x2[n] + j2 * h / 2) - 0.2 * (x4[n] + t2 * h / 2) + x6[n] + b2 * h / 2
        b3 = (-0.1) * (x1[n] + k2 * h / 2 + x5[n] + a2 * h / 2) + 0.2 * (x7[n] + c2 * h / 2)
        c3 = (-0.1) * (x1[n] + k2 * h / 2 + x6[n] + b2 * h / 2 - x8[n] - d2 * h / 2)
        d3 = -0.2 * (x7[n] + c2 * h / 2)

        k4 = 10 * (x2[n] + j3 * h / 2 - x1[n] - k3 * h / 2) + x4[n] + t3 * h / 2
        j4 = 76 * (x1[n] + k3 * h / 2) - (x1[n] + k3 * h / 2) * (x3[n] + r3 * h / 2) + x4[n] + t3 * h / 2
        r4 = (x1[n] + k3 * h / 2) * (x2[n] + j3 * h / 2) - x3[n] - r3 * h / 2 - x4[n] - t3 * h / 2 + x7[n] + c3 * h / 2
        t4 = -3 * (x1[n] + k3 * h / 2 + x2[n] + j3 * h / 2) + x5[n] + a3 * h / 2
        a4 = -(x2[n] + j3 * h / 2) - 0.2 * (x4[n] + t3 * h / 2) + x6[n] + b3 * h / 2
        b4 = (-0.1) * (x1[n] + k3 * h / 2 + x5[n] + a3 * h / 2) + 0.2 * (x7[n] + c3 * h / 2)
        c4 = (-0.1) * (x1[n] + k3 * h / 2 + x6[n] + b3 * h / 2 - x8[n] - d3 * h / 2)
        d4 = -0.2 * (x7[n] + c3 * h / 2)

        x1[n+1] = x1[n] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x2[n+1] = x2[n] + (h / 6) * (j1 + 2 * j2 + 2 * j3 + j4)
        x3[n+1] = x3[n] + (h / 6) * (r1 + 2 * r2 + 2 * r3 + r4)
        x4[n+1] = x4[n] + (h / 6) * (t1 + 2 * t2 + 2 * t3 + t4)
        x5[n+1] = x5[n] + (h / 6) * (a1 + 2 * a2 + 2 * a3 + a4)
        x6[n+1] = x6[n] + (h / 6) * (b1 + 2 * b2 + 2 * b3 + b4)
        x7[n+1] = x7[n] + (h / 6) * (c1 + 2 * c2 + 2 * c3 + c4)
        x8[n+1] = x8[n] + (h / 6) * (d1 + 2 * d2 + 2 * d3 + d4)

    x1 = x1[1:N]
    x1 = np.abs(x1 * 100000000)
    x1 = np.floor(np.mod(x1, 256))

    T = I.copy()
    I = np.uint8(x1.reshape((q, t, c)))
    I = np.bitwise_xor(I, T)

    return I

def decrypt_8dhyperchaos(I, key):
    # Decrypting is essentially reversing the encryption process
    decrypted_image = encryp_8dhyperchaos(I, key)
    return decrypted_image

secret_key = 123  # Replace with your encryption key
#image_path = "D:\\Mine\\Master\\Second Semester\\Softwarwe Engineering\\Assignments\\Encryption\\Image_20240309210952.bmp"

image_path = "D:/Mine/Master/Second Semester/Softwarwe Engineering/Assignments/Encryption/Image_20240322103617.jpg"

#image_path = "D:/Mine/Master/Second Semester/Softwarwe Engineering/Assignments/Encryption/Screenshot 2024-03-02 121154.png"

TT = np.array(Image.open(image_path).convert('RGB'))  # Load the image in RGB mode
X_encrypted = encryp_8dhyperchaos(TT, secret_key)
Image.fromarray(X_encrypted).show()

X_decrypted = decrypt_8dhyperchaos(X_encrypted, secret_key)
Image.fromarray(X_decrypted).show()

mse = calculate_mse(TT, X_decrypted)
print("Mean Squared Error (MSE):", mse)