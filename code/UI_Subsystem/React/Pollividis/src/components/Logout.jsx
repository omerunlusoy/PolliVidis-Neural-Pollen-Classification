import React, {useEffect} from "react";
import { useNavigate } from 'react-router-dom';




const Logout = () => {
    sessionStorage.clear();
    const navigate = useNavigate();
    useEffect(() => {
        navigate("/login");
    },[]);

    return null
};

export default Logout;
