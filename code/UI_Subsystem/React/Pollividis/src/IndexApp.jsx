import {BrowserRouter, Routes, Route} from "react-router-dom";
import App from "./App";
import AboutUs from "./components/AboutUs";
import AnalysisPage from "./components/AnalysisPage";
import Profile from "./components/Profile";
import SignUp from "./components/SignUp";
import Login from "./components/Login";
import PollenMap from "./components/PollenMap";
import AnalyzeSample from "./components/AnalyzeSample";

const IndexApp = () => {
  return (
      <BrowserRouter>
          <Routes>
              <Route path="/" element={<App />} />
              <Route path="/analysis/:id" element={<AnalysisPage />} />
              <Route path="/about-us" element={<AboutUs/>} />
              <Route path="/sign-up" element={<SignUp/>} />
              <Route path="/profile" element={<Profile/>} />
              <Route path="/login" element={<Login/>} />
              <Route path="/map" element={<PollenMap/>} />
              <Route path="/analyze_sample" element={<AnalyzeSample/>} />

          </Routes>
      </BrowserRouter>
  );
};

export default IndexApp;
