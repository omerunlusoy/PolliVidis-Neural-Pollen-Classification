import * as React from 'react';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import {useEffect, useState} from "react";
import styled from "@emotion/styled";
import {Divider, IconButton, Typography} from "@material-ui/core";
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import Geocode from "react-geocode";
import ImageCard from "./ImageCard";
import {getDownloadURL, ref} from "@firebase/storage";
import {storage} from "../firebase";
import Map from "./Map";

const DrawerHeader = styled('div')(({ theme }) => ({
    display: 'flex',
    alignItems: 'center',
    // necessary for content to be below app bar
    justifyContent: 'flex-end',
}));

export default function AnalysisInfoDrawer(props) {


    console.log(props.sample_id)
    let myOpen = true;

    let lat = 39;
    let lng = 38;
    const [address, setAddress] = React.useState("Address unknown");
    const [photo,setPhoto] = useState(null)

    const handleDrawerClose = () => {
        myOpen = false;
        props.parentCallback(myOpen);
    };


    const [analysis, setAnalysis] = useState([])

    useEffect(() => {
        fetch(`http://localhost:8000/api/analysis_get_id/${props.sample_id}/`)
            .then((data) =>  data.json())
            .then((data) => setAnalysis(JSON.parse(data)))
            .then(()=> getPhoto(props.sample_id))
    },[props.sample_id]);

    const getPhoto = async(id) => {

        let fileU = '/files/' + id + '_final.jpg'
        const storageRef = ref(storage,fileU);
        getDownloadURL(storageRef).then((url)=>{
            console.log(url)
            setPhoto(url)
        }).catch((error) => {
            // Handle any errors
            setPhoto("https://images.pexels.com/photos/7319337/pexels-photo-7319337.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=500")
        });
    }


    lat = analysis.location_latitude;
    lng = analysis.location_longitude;

    Geocode.setApiKey("AIzaSyAHlwtPiz1TdtLSNXtladNYvGRtCbzkm6g");
    Geocode.enableDebug();

    Geocode.fromLatLng(lat, lng).then(
        response => {
            setAddress(response.results[0].formatted_address);
            console.log(address);
        },
        error => {
            console.error(error);
        }
    );


    const list = (anchor) => (
        <Box
            sx={{ width: anchor === 'top' || anchor === 'bottom' ? 'auto' : 420 }}
            role="presentation"
        >
            <img style={{
                alignSelf: 'center',
                height: 300,
                width: 400,
            }} src={photo}/>
            <Typography style={{marginBottom:3, marginLeft:10   }} variant="h6"  component="p">
                Location: {analysis.location_latitude}-{analysis.location_longitude}
            </Typography>
            <Typography style={{marginBottom:5, marginLeft:10}} variant="h6"  component="p">
                Date: {analysis.date}
            </Typography>
            <Typography style={{marginBottom:2, marginLeft:10}} variant="h6"  component="p">
                Analysis:
            </Typography>
            <div style={{marginLeft:10}}>
                <pre>
                {analysis.analysis_text}
                </pre>
            </div>

        </Box>
    );

    return (
        <div>
            {
                <React.Fragment key={'left'}>
                    <Drawer
                        variant="persistent"
                        anchor="left"
                        open={props.open}
                    >
                        <DrawerHeader>
                            <IconButton onClick={handleDrawerClose}>
                                <ChevronLeftIcon />
                            </IconButton>
                        </DrawerHeader>
                        <Divider/>
                        {list('left')}
                    </Drawer>
                </React.Fragment>
            }
        </div>
    );
}
