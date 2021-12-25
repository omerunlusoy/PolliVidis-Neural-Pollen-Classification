import React, { useState } from 'react';
import {Container, Grid, makeStyles, Typography} from '@material-ui/core';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import Navbar from "./Navbar";
import Leftbar from "./Leftbar";

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

const SignUp = () => {
    const classes = useStyles();
    // create state variables for each input

    const [fullName, setfullName] = useState('');
    const [appellation, setAppellation] = useState('');
    const [email, setEmail] = useState('');
    const [institution, setInstitution] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = e => {
        e.preventDefault();
        console.log(fullName, appellation, email,institution, password);
    };

    const clearForm = e => {
        e.preventDefault();
        setfullName('');
        setAppellation('');
        setEmail('');
        setInstitution('');
        setPassword('');
    };

    return (
        <div>
            <Navbar />
            <Grid container>
                <Grid item sm={7} xs={10}>
                    <Container className={classes.container}>
                        <Typography style={{marginBottom:10}}variant="h3" component="p">
                            Academic Sign Up
                        </Typography>
                        <form className={classes.root} onSubmit={handleSubmit}>
                            <TextField
                                label="Full Name"
                                variant="filled"
                                required
                                value={fullName}
                                onChange={e => setfullName(e.target.value)}
                            />
                            <TextField
                                label="Appellation"
                                variant="filled"
                                required
                                value={appellation}
                                onChange={e => setAppellation(e.target.value)}
                            />
                            <TextField
                                label="Email"
                                variant="filled"
                                type="email"
                                required
                                value={email}
                                onChange={e => setEmail(e.target.value)}
                            />
                            <TextField
                                label="Institution"
                                variant="filled"
                                required
                                value={institution}
                                onChange={e => setInstitution(e.target.value)}
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
                                <Button variant="contained" onClick={clearForm}>
                                    Clear
                                </Button>
                                <Button type="submit" variant="contained"style={{backgroundColor:'#A6232A', color:'white'}}  onClick={handleSubmit}>
                                    Signup
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

export default SignUp;
