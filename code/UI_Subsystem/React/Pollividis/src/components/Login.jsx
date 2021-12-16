import React, { useState } from 'react';
import {Container, Grid, makeStyles, Typography} from '@material-ui/core';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import Navbar from "./Navbar";
import {Link} from "react-router-dom";

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
    const classes = useStyles();
    // create state variables for each input

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = e => {
        e.preventDefault();
        console.log(email, password);
    };


    return (
        <div>
            <Navbar />
            <Grid container>
                <Grid item sm={7} xs={10}>
                    <Container className={classes.container}>
                        <Typography style={{marginBottom:10}}variant="h3" component="p">
                            Academic Login
                        </Typography>
                        <form className={classes.root} onSubmit={handleSubmit}>
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
                                <Button type="submit" variant="contained" color="primary" onClick={handleSubmit}>
                                    Login
                                </Button>
                            </div>
                        </form>
                    </Container>
                </Grid>
                <Grid item sm={3} className={classes.right}>

                </Grid>
            </Grid>
        </div>
    );
};

export default Login;
