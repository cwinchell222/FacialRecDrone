from djitellopy import Tello
from keras.models import load_model
import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import asyncio
from threading import Thread


takeoff = True
width, height = 480, 320
ref = cv2.imread('ref_img.jpg')
ref_encoding = face_recognition.face_encodings(ref)[0]

def main():
    count = 0
    no_face_count = 0
    frame = 0

    fist_count = 0
    hand_count = 0
    two_hand_count = 0

    model = load_model("Model/keras_model.h5")
    class_names = open("Model/labels.txt", "r").readlines()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    cv2.waitKey(2000)

    tello = Tello()
    tello.connect()
    print(tello.get_battery())
    tello.takeoff()
    cv2.waitKey(1000)
    tello.streamon()
    cv2.waitKey(1000)
    #cv2.waitKey(3000)

    while True:
        img = get_img(tello)
        img = cv2.flip(img, 1)
        #cv2.waitKey(66)

        img_crop = img[int(img.shape[0] - 224):int(img.shape[0]),
                   int(0.25 * img.shape[1]):int((0.25 * img.shape[1]) + 224)]
        img_crop = cv2.resize(img_crop, (224, 224), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb)

        results_hand = hands.process(img_crop_rgb)

        print(f'faces detected: {len(face_locations)}')

        if len(face_locations) == 0:
            no_face_count += 1
            print(f'No face count: {no_face_count}')

        if count == 0:
            print('performing locating')
            locate_thread = Thread(target=locate, daemon=True, kwargs={"tello": tello})
            locate_thread.start()
            locate_thread.join()

        if no_face_count > 14:
            count = 0
            no_face_count = 0

        if face_locations:
            results_face = face_recognition.compare_faces(ref_encoding, face_encodings, tolerance=0.6)
            for each_face in face_locations:
                cv2.rectangle(img, (each_face[3], each_face[0]), (each_face[1], each_face[2]), (255, 0, 0), 1)
            if results_face:
                for i in range(len(results_face)):
                    if results_face[i]:
                        count += 1
                        no_face_count = 0
                        face_match = face_locations[i]
                        cv2.rectangle(img, (face_match[3], face_match[0]), (face_match[1], face_match[2]),
                                      (255, 0, 255), 4)
                        w, h = (face_match[1] - face_match[3]), (face_match[2] - face_match[0])
                        area = w * h

                        center_x = face_match[3] + (w // 2)
                        center_y = face_match[0] + (h // 2)
                        hor_error = (center_x - (width // 2))
                        vert_error = (center_y - (height // 2))

                        if hor_error < -50:
                            print(f'hor_error: {hor_error}. Moving Left')
                            if hor_error < -130:
                                print(f'alot')
                                major_right_thread = Thread(target=major_right_turn, daemon=True, kwargs={"tello": tello})
                                major_right_thread.start()
                                major_right_thread.join()
                            else:
                                print(f'a little')
                                right_thread = Thread(target=right_turn, daemon=True, kwargs={"tello": tello})
                                right_thread.start()
                                right_thread.join()
                        elif hor_error > 50:
                            print(f'hor_error: {hor_error}. Moving Right')
                            if hor_error > 130:
                                print(f'alot')
                                major_left_thread = Thread(target=major_left_turn, daemon=True, kwargs={"tello": tello})
                                major_left_thread.start()
                                major_left_thread.join()
                            else:
                                print(f'a little')
                                left_thread = Thread(target=left_turn, daemon=True, kwargs={"tello": tello})
                                left_thread.start()
                                left_thread.join()
                        elif vert_error > 50:
                            print(f'vert_error: {vert_error}. Moving Down')
                            down_thread = Thread(target=down, daemon=True, kwargs={"tello": tello})
                            down_thread.start()
                            down_thread.join()
                        elif vert_error < -100:
                            print(f'vert_error: {vert_error}. Moving Up')
                            up_thread = Thread(target=up, daemon=True, kwargs={"tello": tello})
                            up_thread.start()
                            up_thread.join()
                        if area > 8000:
                            print(f'Area: {area}. Moving Backwards')
                            if area > 10000:
                                print(f'alot')
                                fast_backward_thread = Thread(target=fast_backward, daemon=True, kwargs={"tello": tello})
                                fast_backward_thread.start()
                                fast_backward_thread.join()
                            else:
                                print(f'a little')
                                backward_thread = Thread(target=backward, daemon=True, kwargs={"tello": tello})
                                backward_thread.start()
                                backward_thread.join()
                        elif 8000 > area > 3000:
                            print(f'Area: {area}. No Movement Required')
                            if results_hand.multi_hand_landmarks:
                                for hand_lms in results_hand.multi_hand_landmarks:
                                    mp_draw.draw_landmarks(img_crop, hand_lms, mp_hands.HAND_CONNECTIONS)

                                img_predict = np.array(img_crop)
                                img_predict = cv2.resize(img_predict, (224, 224,))
                                img_predict_scaled = np.array(img_predict) / 255
                                img_predict_scaled = np.expand_dims(img_predict_scaled, axis=0)

                                prediction = model.predict(img_predict_scaled)
                                index = np.argmax(prediction)
                                class_name = class_names[index]
                                confidence_score = prediction[0][index]
                                confidence_score = int(str(np.round(confidence_score * 100))[:-2])
                                class_name = class_name[2:]
                                print(confidence_score)
                                print(class_name)

                                if index == 0 and confidence_score > 80:
                                    hand_count = 0
                                    two_hand_count = 0
                                    if fist_count > 2:
                                        print('excecuting fist')
                                        fist_thread = Thread(target=fist, daemon=True, kwargs={"tello": tello})
                                        fist_thread.start()
                                        fist_thread.join()
                                        fist_count = 0
                                    else:
                                        fist_count += 1
                                elif index == 1 and confidence_score > 80:
                                    fist_count = 0
                                    two_hand_count = 0
                                    if hand_count > 2:
                                        print('executing hand')
                                        hand_thread = Thread(target=hand, daemon=True, kwargs={"tello": tello})
                                        hand_thread.start()
                                        hand_thread.join()
                                        hand_count = 0
                                    else:
                                        hand_count += 1
                                elif index == 2 and confidence_score > 80:
                                    fist_count = 0
                                    hand_count = 0
                                    if two_hand_count > 2:
                                        print('executing two hand')
                                        two_hand_thread = Thread(target=two_hands, daemon=True, kwargs={"tello": tello})
                                        two_hand_thread.start()
                                        two_hand_thread.join()
                                        two_hand_count = 0
                                    else:
                                        two_hand_count += 1
                        elif 3000 > area > 1500:
                            print(f'Area: {area}. Moving Forward')
                            forward_thread = Thread(target=forward, daemon=True, kwargs={"tello": tello})
                            forward_thread.start()
                            forward_thread.join()
                        elif area < 1500:
                            print(f'Area: {area}. Moving Forward Fast')
                            fast_forward_thread = Thread(target=fast_forward, daemon=True, kwargs={"tello": tello})
                            fast_forward_thread.start()
                            fast_forward_thread.join()
                    else:
                        print('No Matching Face')
                        no_face_count += 1
                        print(f'no object detected {no_face_count} times')
            else:
                print(f'No Faces')
                no_face_count += 1
                print(f'no object detected {no_face_count} times')

        cv2.imshow('crop', img_crop)
        cv2.imshow('img', img)
        frame += 1
        # if frame % 10 == 0:
        #     cv2.destroyAllWindows()
        cv2.waitKey(66)


def get_img(tello):
    my_frame = tello.get_frame_read()
    my_frame = my_frame.frame
    img = cv2.resize(my_frame, (width, height))
    return img


def fist(tello):
    async def main():
        tello.rotate_clockwise(360)
        print('performed FIST gesture')
        # tello.land()

    asyncio.run(main())


def hand(tello):
    async def main():
        tello.land()
        print('performed HAND gesture and landed')

    asyncio.run(main())


def two_hands(tello):
    async def main():
        tello.move_back(75)
        print('performed TWOHANDS gesture')

    asyncio.run(main())


def left_turn(tello):
    async def main():
        tello.rotate_counter_clockwise(5)
        print('turned left 5 degrees')

    asyncio.run(main())


def major_left_turn(tello):
    async def main():
        tello.rotate_counter_clockwise(10)
        print('turned left 10 degrees')

    asyncio.run(main())


def right_turn(tello):
    async def main():
        tello.rotate_clockwise(5)
        print('turned right 5 degrees')

    asyncio.run(main())


def major_right_turn(tello):
    async def main():
        tello.rotate_clockwise(10)
        print('turned right 10 degrees')

    asyncio.run(main())


def forward(tello):
    async def main():
        tello.move_forward(20)
        print('went forward 20cm')

    asyncio.run(main())


def fast_forward(tello):
    async def main():
        tello.move_forward(50)
        print('went forward 50cm')

    asyncio.run(main())


def backward(tello):
    async def main():
        tello.move_back(20)
        print('went back 20cm')

    asyncio.run(main())


def fast_backward(tello):
    async def main():
        tello.move_back(50)
        print('went back 50cm')

    asyncio.run(main())


def up(tello):
    async def main():
        tello.move_up(20)
        print('went up 20cm')

    asyncio.run(main())


def down(tello):
    async def main():
        tello.move_down(20)
        print('went down 20cm')

    asyncio.run(main())


def locate(tello):
    async def main():
        tello.rotate_clockwise(20)
        print('turned right 20 degrees')

    asyncio.run(main())


if __name__ == '__main__':
    main()