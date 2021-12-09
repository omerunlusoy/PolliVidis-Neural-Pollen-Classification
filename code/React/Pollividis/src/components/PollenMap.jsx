import {Card, CardActionArea, CardContent, CardMedia, Container, Grid, makeStyles, Typography} from "@material-ui/core";
import Post from "./Post";
import Navbar from "./Navbar";
import Leftbar from "./Leftbar";
import Add from "./Add";
import SampleImagePreviewCard from "./SampleImagePreviewCard";
import React from "react";
import ImageCard from "./ImageCard";
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
                <Grid item sm={2} xs={2}>
                    <Leftbar />
                </Grid>
                <Grid item sm={10} xs={10}>
                    <Container className={classes.container}>
                        <Typography style={{marginBottom:10}}variant="h3" component="p">
                            Pollen Map
                        </Typography>
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
