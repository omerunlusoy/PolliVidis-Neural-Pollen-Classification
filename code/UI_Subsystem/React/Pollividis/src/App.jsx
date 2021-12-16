import { Grid, makeStyles } from "@material-ui/core";
import Add from "./components/Add";
import Feed from "./components/Feed";
import Leftbar from "./components/Leftbar";
import Navbar from "./components/Navbar";
import Rightbar from "./components/Rightbar";
import AnalyzeSample from "./components/AnalyzeSample";
import SampleImagePreviewCard from "./components/SampleImagePreviewCard";

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
        <AnalyzeSample />
    );
};

export default App;
