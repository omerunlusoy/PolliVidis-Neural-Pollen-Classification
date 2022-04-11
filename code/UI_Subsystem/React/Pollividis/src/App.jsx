import { Grid, makeStyles } from "@material-ui/core";
import PollenMap from "./components/PollenMap";

const useStyles = makeStyles((theme) => ({
    right: {
        [theme.breakpoints.down("sm")]: {
            display: "none",
        },
    },
}));

const App = () => {
    const classes = useStyles();
    return (
        <PollenMap />
    );
};

export default App;
