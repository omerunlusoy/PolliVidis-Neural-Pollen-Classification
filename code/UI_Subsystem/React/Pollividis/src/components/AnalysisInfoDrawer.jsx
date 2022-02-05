import * as React from 'react';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import {useEffect, useState} from "react";
import Button from "@mui/material/Button";
import MenuIcon from "@mui/icons-material/Menu";
import styled from "@emotion/styled";
import {Divider, IconButton} from "@material-ui/core";
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';

const DrawerHeader = styled('div')(({ theme }) => ({
    display: 'flex',
    alignItems: 'center',
    // necessary for content to be below app bar
    justifyContent: 'flex-end',
}));

export default function AnalysisInfoDrawer(props) {

    const [state, setState] = React.useState({
        left: false,
    });

    const [open, setOpen] = React.useState(props.open);

    console.log(props.sample_id)
    let myOpen = true;

    /*
    useEffect(() => {
        setOpen(props.open);
    }, [props.open]) */


    const handleDrawerClose = () => {
        //setOpen(false);
        myOpen = false;
        props.parentCallback(myOpen);
    };

    const toggleDrawer = (anchor, open) => (event) => {
        if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
            return;
        }

        setState({ ...state, [anchor]: open });
    };

    const [analysis, setAnalysis] = useState([])

    useEffect(() => {
        fetch(`http://localhost:8000/api/analysis_get_id/${props.sample_id}/`)
            .then((data) =>  data.json())
            .then((data) => setAnalysis(JSON.parse(data)))
    },[props.sample_id]);


    const list = (anchor) => (
        <Box
            sx={{ width: anchor === 'top' || anchor === 'bottom' ? 'auto' : 250 }}
            role="presentation"
        >
            <p>Location: {analysis.location_latitude}-{analysis.location_longitude}</p>
            <p>Date: {analysis.date}</p>

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
