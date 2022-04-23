import React, {useEffect, useRef, useState} from 'react';
import {Card, CardContent, CardMedia, Container, Grid, makeStyles, Typography} from '@material-ui/core';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import Navbar from "./Navbar";
import { useNavigate } from 'react-router-dom';

import {Link, useHistory} from "react-router-dom";
import ImageCard from "./ImageCard";
import Box from "@mui/material/Box";
import axios from "axios";

const useStyles = makeStyles(theme => ({
    root: {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: theme.spacing(2),

        '& .MuiTextField-root': {
            margin: theme.spacing(1),
            width: '300px',
        },
        '& .MuiButtonBase-root': {
            margin: theme.spacing(2),
        },
    },
    container: {
        paddingTop: theme.spacing(10),
    },
}));

const Login = ({ handleClose }) => {

    const navigate = useNavigate();

    const classes = useStyles();
    // create state variables for each input

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const [user_logged_in, setUser_logged_in] = useState(false)


    const handleSubmit = e => {
        e.preventDefault();
        var info = email + "~" + password;


        fetch(`http://localhost:8000/api/login/${info}/`)
            .then((data) => {
                if(!data.ok) throw new Error(data.status);
                else return data.json();
            })
            .then( (data) => {sessionStorage.setItem('academic_id',JSON.stringify(data))
                console.log(data)
                setUser_logged_in(true)
                console.log(user_logged_in)
            }
            )
            // .then((data) =>)
            //.then()
            //.then( navigate("/profile") )
            .catch( error => {
                //this.setState({ errorMessage: error.toString() });
                console.log('There was an error!', error);
                alert("Please check the login information")
            });


    };

    useEffect(() => {

        if (user_logged_in) {
            navigate("/profile")
        }}, [user_logged_in]);


    return (
        <div>
            <Navbar />
                <Grid container>
                            <Card sx={{flexDirection: 'row'}}>
                                <Grid item sm={7} >
                                <Box sx={{ display: 'flex', flexDirection: 'row' }}>
                                    <CardMedia
                                        component="img"
                                        image="/welcome_pollividis.png"
                                        alt="Live from space album cover"
                                    />
                                    <form style={{marginLeft: 85}} className={classes.root} onSubmit={handleSubmit}>
                                        <TextField
                                            label="Email"
                                            variant="filled"
                                            type="email"
                                            required
                                            value={email}
                                            onChange={e => setEmail(e.target.value)}
                                        />
                                        <TextField
                                            label="Password"
                                            variant="filled"
                                            type="password"
                                            required
                                            value={password}
                                            onChange={e => setPassword(e.target.value)}
                                        />
                                        <div>
                                            <Link style={{ textDecoration:'none'}} to="/sign-up"><Button variant="contained" >Sign Up</Button></Link>
                                            <Button type="submit" variant="contained" style={{backgroundColor:'#A6232A', color:'white'}}  >
                                                Login
                                            </Button>
                                        </div>
                                    </form>
                                </Box>
                                </Grid>
                            </Card>
                </Grid>
        </div>
    );
};

export default Login;
