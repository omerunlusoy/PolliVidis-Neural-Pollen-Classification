import React, { useState } from 'react';
import {Container, Grid, makeStyles, Typography} from '@material-ui/core';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import Navbar from "./Navbar";
import Leftbar from "./Leftbar";
import axios from 'axios';
import Form from "react-validation/build/form";
import Input from "react-validation/build/input";
import CheckButton from "react-validation/build/button";
import { isEmail } from "validator";



const required = value => {
    if (!value) {
        return (
            <div className="alert alert-danger" role="alert">
                This field is required!
            </div>
        );
    }
};

const vemail = value => {
    if (!isEmail(value)) {
        return (
            <div className="alert alert-danger" role="alert">
                This is not a valid email.
            </div>
        );
    }
};

const vusername = value => {
    if (value.length < 3 || value.length > 20) {
        return (
            <div className="alert alert-danger" role="alert">
                The username must be between 3 and 20 characters.
            </div>
        );
    }
};

//will change
const vpassword = value => {
    if (value.length < 6 || value.length > 40) {
        return (
            <div className="alert alert-danger" role="alert">
                The password must be between 6 and 40 characters.
            </div>
        );
    }
};


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

    const [message, setMessage] = useState('');



    const handleSubmit = e => {
        e.preventDefault();


        console.log(fullName, appellation, email,institution, password);

        let sampleObject = new FormData(); // creates a new FormData object

        const myObject = {
            academic_id: 0,
            name: fullName,
            surname: fullName,
            appellation: appellation,
            institution: institution,
            job_title: "a",
            email: email,
            password: password,
            photo: "",
            reserch_gate_link:"link"
        };

        sampleObject.append("academic_id",myObject.academic_id);
        sampleObject.append("name", myObject.name);
        sampleObject.append("surname", myObject.surname);// add your file to form data
        sampleObject.append("appellation", myObject.appellation);// add your file to form data
        sampleObject.append("institution", myObject.institution);// add your file to form data
        sampleObject.append("job_title", myObject.job_title);// add your file to form data
        sampleObject.append("email", myObject.email);// add your file to form data
        sampleObject.append("password", myObject.password);// add your file to form data
        sampleObject.append("photo", myObject.photo);// add your file to form data
        sampleObject.append("research_gate_link", myObject.reserch_gate_link);// add your file to form data

        axios
            .post('http://127.0.0.1:8000/api/sign-up/', sampleObject)
            .then(response => {
                setMessage(response.data.message)
            })
            .catch(error => {
                const resMessage =
                    (error.response &&
                        error.response.data &&
                        error.response.data.message) ||
                    error.message ||
                    error.toString();

            })


    };

    const clearForm = e => {
        e.preventDefault();
        setfullName('');
        setAppellation('');
        setEmail('');
        setInstitution('');
        setPassword('');
    };


    /*
    return (
        <div className="col-md-12">
            <div className="card card-container">
                <Form
                    onSubmit={handleSubmit}
                    ref={c => {
                        this.form = c;
                    }}
                >
                    {!this.state.successful && (
                        <div>
                            <div className="form-group">
                                <label htmlFor="username">Name</label>
                                <Input
                                    type="text"
                                    className="form-control"
                                    name="username"
                                    value={fullName}
                                    onChange={e => setfullName(e.target.value)}
                                    validations={[required, vusername]}
                                />
                            </div>
                            <div className="form-group">
                                <label htmlFor="email">Email</label>
                                <Input
                                    type="text"
                                    className="form-control"
                                    name="email"
                                    value={email}
                                    onChange={e => setEmail(e.target.value)}
                                    validations={[required, vemail]}
                                />
                            </div>
                            <div className="form-group">
                                <label htmlFor="password">Password</label>
                                <Input
                                    type="password"
                                    className="form-control"
                                    name="password"
                                    value={password}
                                    onChange={e => setPassword(e.target.value)}
                                    validations={[required, vpassword]}
                                />
                            </div>
                            <div className="form-group">
                                <button className="btn btn-primary btn-block">Sign Up</button>
                            </div>
                        </div>
                    )}
                    {this.state.message && (
                        <div className="form-group">
                            <div
                                className={
                                    this.state.successful
                                        ? "alert alert-success"
                                        : "alert alert-danger"
                                }
                                role="alert"
                            >
                                {this.state.message}
                            </div>
                        </div>
                    )}
                    <CheckButton
                        style={{ display: "none" }}
                        ref={c => {
                            this.checkBtn = c;
                        }}
                    />
                </Form>
            </div>
        </div>
    );*/

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
