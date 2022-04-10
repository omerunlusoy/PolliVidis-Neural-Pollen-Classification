import { initializeApp } from "firebase/app";
import { getStorage } from "firebase/storage";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyDCtmcYxx7-MqKH3CVH8MCW-XwtVcfFO3Y",
    authDomain: "fir-react1-70dd6.firebaseapp.com",
    projectId: "fir-react1-70dd6",
    storageBucket: "fir-react1-70dd6.appspot.com",
    messagingSenderId: "1025002303663",
    appId: "1:1025002303663:web:c99c07e49fdf21e149e45a",
    measurementId: "G-YH1V8NZCSF"
};

export const app= initializeApp(firebaseConfig)
export const storage = getStorage(app)
