import {BrowserRouter, Routes, Route} from "react-router-dom";
import App from "./App";
import AboutUs from "./components/AboutUs";
import Logout from "./components/Logout";

import AnalysisPage from "./components/AnalysisPage";
import Profile from "./components/Profile";
import SignUp from "./components/SignUp";
import Login from "./components/Login";
import PollenMap from "./components/PollenMap";
import AnalyzeSample from "./components/AnalyzeSample";
import SendFeedback from "./components/SendFeedback";
import HowItWorks from "./components/HowItWorks";
import PreviousAnalyses from "./components/PreviousAnalyses";

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
              <Route path="/send_feedback" element={<SendFeedback/>} />
              <Route path="/how_pollividis_works" element={<HowItWorks/>} />
              <Route path="/previous_analyses" element={<PreviousAnalyses/>} />
              <Route path="/logout" element={<Logout/>} />

          </Routes>
      </BrowserRouter>
  );
};

export default IndexApp;
