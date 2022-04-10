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
    const [address, setAddress] = React.useState("");
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

        let fileU = '/files/' + id
        const storageRef = ref(storage,fileU);
        getDownloadURL(storageRef).then((url)=>{
            console.log(url)
            setPhoto(url)
        }).catch((error) => {
            // Handle any errors
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
            sx={{ width: anchor === 'top' || anchor === 'bottom' ? 'auto' : 250 }}
            role="presentation"
        >
            <ImageCard img={photo}/>
            <Typography style={{marginBottom:3}} variant="h6"  component="p">
                Location: {address}
            </Typography>
            <Typography style={{marginBottom:5}} variant="h6"  component="p">
                Date: {analysis.date}
            </Typography>
            <Typography style={{marginBottom:2}} variant="h6"  component="p">
                Analysis:
            </Typography>
            <Typography style={{marginBottom:2}} variant="h5"  component="p">
                {analysis.analysis_text}
            </Typography>

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
