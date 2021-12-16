import {Card, CardActionArea, CardContent, CardMedia, Container, Grid, makeStyles, Typography} from "@material-ui/core";
import Navbar from "./Navbar";
import React from "react";
import Map from "./Map";

const useStyles = makeStyles((theme) => ({
    container: {
        paddingTop: theme.spacing(10),
    },
}));

const PollenMap = () => {
    const classes = useStyles();
    return (
        <div>
            <Navbar />
            <Grid container>
                <Grid item sm={12} xs={10}>
                    <Container className={classes.container}>
                        <Card>
                            <CardActionArea>
                                <CardContent>
                                    <Map/>
                                </CardContent>
                            </CardActionArea>
                        </Card>

                    </Container>
                </Grid>
                <Grid item sm={3} className={classes.right}>

                </Grid>
            </Grid>
        </div>
    );
};

export default PollenMap;
